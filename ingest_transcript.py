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

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is missing from your .env file")


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


def main():
    transcript = get_transcript(VIDEO_ID)
    metadata = {
        "video_id": VIDEO_ID,
        "source": f"https://www.youtube.com/watch?v={VIDEO_ID}",
        "language": "en",
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.create_documents([transcript], metadatas=[metadata])

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(DATABASE_DIR),
        collection_name=COLLECTION_NAME,
    )

    chunk_ids = [f"{VIDEO_ID}_chunk_{index}" for index in range(len(chunks))]
    existing_ids = set(vector_store.get(ids=chunk_ids)["ids"])
    new_chunks = []
    new_ids = []

    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        vector_store.add_documents(documents=new_chunks, ids=new_ids)
        print(f"Added {len(new_chunks)} new chunks.")
    else:
        print("All chunks already exist. Nothing added.")

    print("Total records in collection:", vector_store._collection.count())


if __name__ == "__main__":
    main()
