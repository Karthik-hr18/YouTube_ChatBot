import os
from pathlib import Path

from dotenv import load_dotenv
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
COLLECTION_NAME = "youtube_transcripts_timestamped"
VIDEO_ID = "RE2c_eH0Sq4"

load_dotenv(BASE_DIR / ".env")


def format_timestamp(seconds):
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def get_transcript(video_id):
    try:
        fetched_transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"],
        )
        transcript_list = fetched_transcript.to_raw_data()
        transcript = "\n".join(
            f"[{format_timestamp(item['start'])}] {item['text']}"
            for item in transcript_list
        )

        if not transcript.strip():
            raise ValueError("The transcript is empty")

        return transcript
    except TranscriptsDisabled as error:
        raise RuntimeError("Captions are disabled for this video") from error
    except Exception as error:
        raise RuntimeError(f"Could not fetch transcript: {error}") from error


def get_vector_store():
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is missing from your .env file")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(DATABASE_DIR),
        collection_name=COLLECTION_NAME,
    )


def ingest_video(video_id):
    """Fetch and index a video's transcript unless it is already indexed."""
    if not video_id or not video_id.strip():
        raise ValueError("video_id must not be empty")

    vector_store = get_vector_store()
    existing_video = vector_store.get(
        where={"video_id": video_id},
        include=["metadatas"],
    )
    if existing_video["ids"]:
        return {
            "video_id": video_id,
            "indexed": False,
            "message": "Video is already indexed.",
            "chunk_count": len(existing_video["ids"]),
        }

    transcript = get_transcript(video_id)
    metadata = {
        "video_id": video_id,
        "source": f"https://www.youtube.com/watch?v={video_id}",
        "language": "en",
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.create_documents([transcript], metadatas=[metadata])

    chunk_ids = [f"{video_id}_chunk_{index}" for index in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=chunk_ids)

    return {
        "video_id": video_id,
        "indexed": True,
        "message": "Transcript indexed successfully.",
        "chunk_count": len(chunks),
    }


def main():
    result = ingest_video(VIDEO_ID)
    print(result["message"])
    print(f"Chunks: {result['chunk_count']}")


if __name__ == "__main__":
    main()
