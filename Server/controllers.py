import logging
import time
from threading import Lock
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from RAG_Service.ingest_transcript import ingest_video
from RAG_Service.lang_chain import answer_question
from RAG_Service.video_utils import extract_video_id


logger = logging.getLogger(__name__)
router = APIRouter()
jobs = {}
jobs_lock = Lock()
JOB_TTL_SECONDS = 30 * 60
MAX_JOBS = 500


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


def run_question_job(job_id: str, video_id: str, question: str, refresh: bool):
    with jobs_lock:
        jobs[job_id]["status"] = "processing"

    try:
        result = handle_question(video_id, question, refresh)
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "completed",
                    "result": result,
                }
            )
    except ValueError as error:
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "failed",
                    "error": str(error),
                    "error_status": 400,
                }
            )
    except RuntimeError as error:
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "failed",
                    "error": str(error),
                    "error_status": 502,
                }
            )
    except Exception:
        logger.exception("Unexpected error while processing job %s", job_id)
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": "failed",
                    "error": "An unexpected server error occurred.",
                    "error_status": 500,
                }
            )


def prune_jobs():
    cutoff = time.monotonic() - JOB_TTL_SECONDS
    expired = [
        job_id
        for job_id, job in jobs.items()
        if job["created_at"] < cutoff
        and job["status"] in {"completed", "failed"}
    ]
    for job_id in expired:
        del jobs[job_id]

    if len(jobs) > MAX_JOBS:
        terminal_jobs = sorted(
            (
                job_id
                for job_id, job in jobs.items()
                if job["status"] in {"completed", "failed"}
            ),
            key=lambda job_id: jobs[job_id]["created_at"],
        )
        for job_id in terminal_jobs[: max(0, len(jobs) - MAX_JOBS)]:
            del jobs[job_id]


@router.post("/ask", status_code=status.HTTP_202_ACCEPTED)
def ask(request: AskRequest, background_tasks: BackgroundTasks):
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
    with jobs_lock:
        prune_jobs()
        jobs[job_id] = {
            "status": "queued",
            "created_at": time.monotonic(),
        }

    background_tasks.add_task(
        run_question_job,
        job_id,
        video_id,
        question,
        request.refresh,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/ask/{job_id}",
    }


@router.get("/ask/{job_id}")
def get_job_status(job_id: str):
    with jobs_lock:
        prune_jobs()
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Job not found or expired.",
            )
        response = {"job_id": job_id, "status": job["status"]}
        if job["status"] == "completed":
            response["result"] = job["result"]
        elif job["status"] == "failed":
            response["error"] = job["error"]
            response["error_status"] = job["error_status"]
        return response
