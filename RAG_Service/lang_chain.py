import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
COLLECTION_NAME = "youtube_transcripts_timestamped"

load_dotenv(BASE_DIR.parent / ".env")

prompt = PromptTemplate(
    template="""
Answer using only the transcript chunks below.

Rules:
- Support every factual claim with a timestamp citation from the chunks.
- If the chunks don't contain enough evidence, say the transcript does not
  provide enough information.
- Do not guess speaker identities or invent timestamps.
- For questions with multiple parts, answer each part separately and cite its
  evidence.
- If the question is ambiguous, say what is unclear.

Transcript chunks:
{context}

Question: {question}

Answer:
""",
    input_variables=["context", "question"],
)


def format_docs(retrieved_docs):
    return "\n\n--- Retrieved chunk ---\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown source')}\n"
        f"{doc.page_content}"
        for doc in retrieved_docs
    )


def get_vector_store():
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is missing from your .env file")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(DATABASE_DIR),
        collection_name=COLLECTION_NAME,
    )


def print_retrieved_docs(retrieved_docs):
    for index, doc in enumerate(retrieved_docs, start=1):
        timestamps = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", doc.page_content)
        print(f"\n--- Retrieved chunk {index} ---")
        print(f"Video ID: {doc.metadata.get('video_id', 'Unknown')}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Timestamps: {', '.join(timestamps) if timestamps else 'None found'}")
        print("Text:")
        print(doc.page_content)


def get_available_videos(vector_store):
    result = vector_store.get(include=["metadatas"])
    videos = {}

    for metadata in result["metadatas"] or []:
        video_id = metadata.get("video_id")
        if video_id:
            videos[video_id] = metadata.get(
                "source",
                f"https://www.youtube.com/watch?v={video_id}",
            )

    return sorted(videos.items())


def select_video(videos):
    if not videos:
        raise RuntimeError("No indexed videos with video_id metadata were found.")

    print("Choose which video to ask about:")
    for index, (video_id, source) in enumerate(videos, start=1):
        print(f"{index}. {video_id} — {source}")

    selection = input("Enter a number or video ID: ").strip()
    if selection.isdigit():
        index = int(selection) - 1
        if 0 <= index < len(videos):
            return videos[index][0]

    for video_id, _ in videos:
        if selection == video_id:
            return video_id

    raise ValueError("Choose a listed number or enter one of the listed video IDs.")


def answer_question(video_id, question):
    """Answer a question using only transcript chunks for the requested video."""
    if not video_id or not video_id.strip():
        raise ValueError("video_id must not be empty")
    if not question or not question.strip():
        raise ValueError("question must not be empty")

    vector_store = get_vector_store()
    indexed_video = vector_store.get(
        where={"video_id": video_id},
        include=["metadatas"],
    )
    if not indexed_video["ids"]:
        raise ValueError(f"Video {video_id} has not been indexed.")

    retrieved_docs = vector_store.similarity_search(
        question,
        k=4,
        filter={"video_id": video_id},
    )
    if not retrieved_docs:
        raise RuntimeError(
            f"No matching transcript chunks found for video {video_id}."
        )

    answer_chain = prompt | ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    ) | StrOutputParser()
    answer = answer_chain.invoke(
        {
            "context": format_docs(retrieved_docs),
            "question": question,
        }
    )

    citations = []
    seen_citations = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source")
        for match in re.finditer(r"\[(\d{2}):(\d{2}):(\d{2})\]", doc.page_content):
            timestamp = match.group(0)[1:-1]
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = int(match.group(3))
            total_seconds = hours * 3600 + minutes * 60 + seconds
            citation_key = (source, total_seconds)
            if source and citation_key not in seen_citations:
                seen_citations.add(citation_key)
                citations.append(
                    {
                        "timestamp": timestamp,
                        "source": source,
                        "url": f"{source}&t={total_seconds}s",
                    }
                )

    return {
        "video_id": video_id,
        "answer": answer,
        "citations": citations,
    }


def main():
    vector_store = get_vector_store()
    videos = get_available_videos(vector_store)
    selected_video_id = select_video(videos)
    question = input("Ask a question about the video: ").strip()
    result = answer_question(selected_video_id, question)
    print("\nAnswer:")
    print(result["answer"])
    for citation in result["citations"]:
        print(f"{citation['timestamp']}: {citation['url']}")


if __name__ == "__main__":
    main()
