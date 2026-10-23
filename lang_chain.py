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

load_dotenv(BASE_DIR / ".env")

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is missing from your .env file")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=str(DATABASE_DIR),
    collection_name=COLLECTION_NAME,
)

if vector_store._collection.count() == 0:
    raise RuntimeError(
        "No transcript chunks found. Run ingest_transcript.py first."
    )

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

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


answer_chain = prompt | llm | StrOutputParser()


def print_retrieved_docs(retrieved_docs):
    for index, doc in enumerate(retrieved_docs, start=1):
        timestamps = re.findall(r"\[(\d{2}:\d{2}:\d{2})\]", doc.page_content)
        print(f"\n--- Retrieved chunk {index} ---")
        print(f"Video ID: {doc.metadata.get('video_id', 'Unknown')}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Timestamps: {', '.join(timestamps) if timestamps else 'None found'}")
        print("Text:")
        print(doc.page_content)


def get_available_videos():
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


def main():
    videos = get_available_videos()
    selected_video_id = select_video(videos)

    question = input("Ask a question about the video: ").strip()
    if not question:
        raise ValueError("Please enter a question.")

    retrieved_docs = vector_store.similarity_search(
        question,
        k=4,
        filter={"video_id": selected_video_id},
    )
    if not retrieved_docs:
        raise RuntimeError(
            f"No matching transcript chunks found for video {selected_video_id}."
        )

    # print_retrieved_docs(retrieved_docs)

    answer = answer_chain.invoke(
        {
            "context": format_docs(retrieved_docs),
            "question": question,
        }
    )
    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
