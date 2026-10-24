import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from RAG_Service.video_utils import extract_video_id


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
DATABASE_DIR = Path(
    os.getenv("RAG_DATABASE_DIR", str(BASE_DIR / "database"))
).expanduser()
COLLECTION_NAME = "youtube_transcripts_v2"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")

ANSWER_WITH_CITATIONS = (
    "The retrieved transcript evidence is insufficient to provide a "
    "reliably cited answer."
)

prompt = PromptTemplate(
    template="""Answer the question using only the transcript chunks below.

Requirements:
- Cite every factual claim with one or more exact chunk labels such as [S1].
- Use only labels present in the transcript chunks; never invent a label.
- If the chunks do not support an answer, say the transcript does not provide
  enough information.
- Do not invent speakers, facts, or timestamps.
- Answer each part of a multi-part question and cite the supporting chunks.
- If the question is ambiguous, state what is unclear.

Transcript chunks:
{context}

Question: {question}

Answer:
""",
    input_variables=["context", "question"],
)


def get_vector_store():
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is missing from the project .env file")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(DATABASE_DIR),
        collection_name=COLLECTION_NAME,
    )


def format_docs(retrieved_docs):
    formatted_chunks = []
    for index, doc in enumerate(retrieved_docs, start=1):
        start = doc.metadata.get("timestamp_start", "unknown")
        end = doc.metadata.get("timestamp_end", start)
        formatted_chunks.append(
            f"[S{index}] Source: {doc.metadata.get('source', 'Unknown source')}\n"
            f"Timestamp range: {start} - {end}\n"
            f"{doc.page_content}"
        )
    return "\n\n--- Retrieved chunk boundary ---\n\n".join(formatted_chunks)


def split_independent_questions(question):
    """Split explicit questions and clear coordinated question clauses."""
    question = question.strip()
    if not question:
        return []

    parts = re.split(r"\?\s*(?=\S)|[\r\n]+|;\s*", question)
    questions = []
    question_starters = (
        r"who|what|when|where|why|how|which|whose|whom|"
        r"can|could|did|do|does|is|are|was|were|will|would|"
        r"has|have|had|should"
    )
    for part in parts:
        part = part.strip(" \t-•?0123456789.)")
        part = re.sub(r"^(?:and|also)\s+", "", part, flags=re.IGNORECASE)
        if not part:
            continue
        coordinated = re.split(
            rf"\s+(?:and|also)\s+(?=(?:{question_starters})\b)",
            part,
            flags=re.IGNORECASE,
        )
        questions.extend(piece.strip() for piece in coordinated if piece.strip())

    return questions or [question]


def is_broad_question(question):
    normalized = question.lower()
    broad_phrases = (
        "what is this video about",
        "what's this video about",
        "what is the video about",
        "summarize this video",
        "summary of this video",
        "give me an overview",
        "main topic",
        "overall message",
        "main idea",
    )
    return any(phrase in normalized for phrase in broad_phrases)


def get_video_overview_docs(vector_store, video_id, limit=8):
    """Choose chunks spread across the transcript for video-level questions."""
    stored = vector_store.get(
        where={"video_id": video_id},
        include=["documents", "metadatas"],
    )
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    chunks = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(documents, metadatas)
        if text
    ]
    chunks.sort(key=lambda doc: doc.metadata.get("chunk_index", 0))
    if len(chunks) <= limit:
        return chunks

    last_index = len(chunks) - 1
    selected_indices = {
        round(index * last_index / (limit - 1))
        for index in range(limit)
    } if limit > 1 else {last_index // 2}
    return [chunks[index] for index in sorted(selected_indices)]


def attach_validated_citations(answer, retrieved_docs):
    """Replace valid source labels with timestamps and return matching links."""
    labels = {
        f"S{index}": doc.metadata
        for index, doc in enumerate(retrieved_docs, start=1)
    }
    citation_tokens = re.findall(r"\[S[^\]]*\]", answer)
    cited_labels = [token[1:-1] for token in citation_tokens]
    if (
        not cited_labels
        or any(label not in labels for label in cited_labels)
        or re.search(r"\[\d{2}:\d{2}:\d{2}\]", answer)
    ):
        return ANSWER_WITH_CITATIONS, []

    sentences = [
        sentence.strip(" \t-*0123456789.)")
        for sentence in re.split(r"(?<=[.!?])\s+|\r?\n+", answer)
    ]
    for sentence in sentences:
        if not sentence or sentence.endswith(":"):
            continue
        if not re.search(r"\[S\d+\]", sentence):
            return ANSWER_WITH_CITATIONS, []

    citations = []
    seen_labels = set()
    for label in cited_labels:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        metadata = labels[label]
        source = metadata.get("source")
        timestamp_start = metadata.get("timestamp_start")
        timestamp_end = metadata.get("timestamp_end", timestamp_start)
        start_seconds = metadata.get("timestamp_start_seconds")
        if not source or not timestamp_start or start_seconds is None:
            return ANSWER_WITH_CITATIONS, []
        citations.append(
            {
                "timestamp": timestamp_start,
                "timestamp_end": timestamp_end,
                "source": source,
                "url": f"{source}&t={int(start_seconds)}s",
            }
        )

    rendered_answer = re.sub(
        r"\[(S\d+)\]",
        lambda match: f"[{labels[match.group(1)]['timestamp_start']}]",
        answer,
    )
    return rendered_answer, citations


def answer_question(video_id, question):
    """Answer one or more questions using evidence from a single video."""
    if not video_id or not video_id.strip():
        raise ValueError("video_id must not be empty")
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    video_id = extract_video_id(video_id)
    question = question.strip()
    vector_store = get_vector_store()
    indexed_video = vector_store.get(
        where={"video_id": video_id},
        include=["metadatas"],
        limit=1,
    )
    if not indexed_video["ids"]:
        raise ValueError(f"Video {video_id} has not been indexed.")

    answer_chain = prompt | ChatGoogleGenerativeAI(model=CHAT_MODEL) | StrOutputParser()
    question_answers = []
    all_citations = []
    seen_citations = set()

    for subquestion in split_independent_questions(question):
        if is_broad_question(subquestion):
            retrieved_docs = get_video_overview_docs(vector_store, video_id)
        else:
            retrieved_docs = vector_store.similarity_search(
                subquestion,
                k=4,
                filter={"video_id": video_id},
            )

        if not retrieved_docs:
            answer = ANSWER_WITH_CITATIONS
            citations = []
        else:
            raw_answer = answer_chain.invoke(
                {
                    "context": format_docs(retrieved_docs),
                    "question": subquestion,
                }
            )
            answer, citations = attach_validated_citations(
                raw_answer,
                retrieved_docs,
            )

        question_answers.append(
            {
                "question": subquestion,
                "answer": answer,
                "citations": citations,
            }
        )
        for citation in citations:
            key = citation["url"]
            if key not in seen_citations:
                seen_citations.add(key)
                all_citations.append(citation)

    rendered_answers = "\n\n".join(
        f"{index}. {item['question']}\n{item['answer']}"
        for index, item in enumerate(question_answers, start=1)
    )

    return {
        "video_id": video_id,
        "answer": rendered_answers,
        "answers": question_answers,
        "citations": all_citations,
    }


def main():
    video_id = input("YouTube video URL or ID: ").strip()
    question = input("Ask a question about the video: ").strip()
    result = answer_question(video_id, question)
    print("\nAnswer:")
    print(result["answer"])
    for citation in result["citations"]:
        print(f"{citation['timestamp']}: {citation['url']}")


if __name__ == "__main__":
    main()
