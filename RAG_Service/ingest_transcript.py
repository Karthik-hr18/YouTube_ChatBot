import os
import hashlib
import re
from pathlib import Path

from dotenv import load_dotenv
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from RAG_Service.video_utils import extract_video_id, format_timestamp


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
DATABASE_DIR = Path(
    os.getenv("RAG_DATABASE_DIR", str(BASE_DIR / "database"))
).expanduser()
COLLECTION_NAME = "youtube_transcripts_v2"
VIDEO_ID = "RE2c_eH0Sq4"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
INDEX_SCHEMA_VERSION = 2


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


def ingest_video(video_id, refresh=False):
    """Index a video's transcript; refresh replaces old chunks after successful indexing."""
    if not video_id or not video_id.strip():
        raise ValueError("video_id must not be empty")
    video_id = extract_video_id(video_id)

    vector_store = get_vector_store()
    existing_video = vector_store.get(
        where={"video_id": video_id},
        include=["metadatas"],
    )
    existing_ids = existing_video["ids"]
    if existing_ids and not refresh:
        return {
            "video_id": video_id,
            "indexed": False,
            "message": "Video is already indexed.",
            "chunk_count": len(existing_ids),
        }

    transcript = get_transcript(video_id)
    index_signature = (
        f"{transcript}\n"
        f"chunk_size={CHUNK_SIZE}\n"
        f"chunk_overlap={CHUNK_OVERLAP}\n"
        f"schema={INDEX_SCHEMA_VERSION}"
    )
    index_hash = hashlib.sha256(index_signature.encode("utf-8")).hexdigest()
    existing_hashes = {
        metadata.get("index_hash")
        for metadata in (existing_video["metadatas"] or [])
        if metadata
    }
    if existing_ids and existing_hashes == {index_hash}:
        return {
            "video_id": video_id,
            "indexed": False,
            "message": "Transcript is unchanged; existing chunks were kept.",
            "chunk_count": len(existing_ids),
        }

    metadata = {
        "video_id": video_id,
        "source": f"https://www.youtube.com/watch?v={video_id}",
        "language": "en",
        "index_hash": index_hash,
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([transcript], metadatas=[metadata])
    if not chunks:
        raise RuntimeError("The transcript did not produce any indexable chunks.")

    chunk_ids = []
    for index, chunk in enumerate(chunks):
        timestamps = []
        for match in re.finditer(
            r"\[(\d{2}):(\d{2}):(\d{2})\]",
            chunk.page_content,
        ):
            hours, minutes, seconds = map(int, match.groups())
            timestamps.append(hours * 3600 + minutes * 60 + seconds)
        chunk.metadata["chunk_index"] = index
        if timestamps:
            start_seconds = timestamps[0]
            end_seconds = timestamps[-1]
            chunk.metadata.update(
                {
                    "timestamp_start": format_timestamp(start_seconds),
                    "timestamp_end": format_timestamp(end_seconds),
                    "timestamp_start_seconds": start_seconds,
                    "timestamp_end_seconds": end_seconds,
                }
            )
        chunk_ids.append(f"{video_id}_{index_hash[:16]}_chunk_{index}")

    vector_store.add_documents(documents=chunks, ids=chunk_ids)
    obsolete_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in chunk_ids]
    if obsolete_ids:
        vector_store.delete(ids=obsolete_ids)

    return {
        "video_id": video_id,
        "indexed": True,
        "message": (
            "Transcript indexed successfully."
            if not existing_ids
            else "Transcript refreshed successfully."
        ),
        "chunk_count": len(chunks),
    }


def main():
    result = ingest_video(VIDEO_ID)
    print(result["message"])
    print(f"Chunks: {result['chunk_count']}")


if __name__ == "__main__":
    main()
