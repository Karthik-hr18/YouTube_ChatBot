import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from RAG_Service.ingest_transcript import ingest_video
from RAG_Service.lang_chain import answer_question
from RAG_Service.video_utils import extract_video_id
from Server.job_store import (
    claim_next_job,
    complete_job,
    enqueue_job,
    fail_job,
    get_job,
)


logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    video_url: str = Field(
        min_length=1,
        max_length=2048,
        description="YouTube URL or 11-character video ID",
    )
    question: str = Field(min_length=1, max_length=2000)
    refresh: bool = False


def handle_question(video_id: str, question: str, refresh: bool = False) -> dict:
    ingest_video(video_id, refresh=refresh)
    return answer_question(video_id, question)


def run_question_job(job):
    job_id = job["job_id"]
    try:
        result = handle_question(
            job["video_id"],
            job["question"],
            job["refresh"],
        )
        complete_job(job_id, result)
    except ValueError as error:
        fail_job(job_id, str(error), 400)
    except RuntimeError as error:
        fail_job(job_id, str(error), 502)
    except Exception:
        logger.exception("Unexpected error while processing job %s", job_id)
        fail_job(job_id, "An unexpected server error occurred.", 500)


async def job_worker():
    while True:
        job = await asyncio.to_thread(claim_next_job)
        if job is None:
            await asyncio.sleep(1)
            continue
        await asyncio.to_thread(run_question_job, job)


@router.post("/ask", status_code=status.HTTP_202_ACCEPTED)
def ask(request: AskRequest):
    video_url = request.video_url.strip()
    question = request.question.strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="A YouTube URL or video ID is required.")
    if not question:
        raise HTTPException(status_code=400, detail="A question is required.")

    try:
        video_id = extract_video_id(video_url)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    job_id = str(uuid4())
    enqueue_job(job_id, video_id, question, request.refresh)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/ask/{job_id}",
    }


@router.get("/ask/{job_id}")
def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found or expired.",
        )
    return job
