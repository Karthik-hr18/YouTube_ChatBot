import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
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
    """Answer a question using only indexed transcript chunks for this video."""
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

    retrieved_docs = vector_store.similarity_search(
        question,
        k=4,
        filter={"video_id": video_id},
    )
    if not retrieved_docs:
        return {
            "video_id": video_id,
            "answer": ANSWER_WITH_CITATIONS,
            "citations": [],
        }

    answer_chain = prompt | ChatGoogleGenerativeAI(model=CHAT_MODEL) | StrOutputParser()
    raw_answer = answer_chain.invoke(
        {
            "context": format_docs(retrieved_docs),
            "question": question,
        }
    )
    answer, citations = attach_validated_citations(raw_answer, retrieved_docs)

    return {
        "video_id": video_id,
        "answer": answer,
        "citations": citations,
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
