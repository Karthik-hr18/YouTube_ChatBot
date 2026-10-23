from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from RAG_Service.ingest_transcript import ingest_video
from RAG_Service.lang_chain import answer_question


router = APIRouter()


class AskRequest(BaseModel):
    video_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


def handle_question(video_id: str, question: str) -> dict:
    ingest_video(video_id)
    return answer_question(video_id, question)


@router.post("/ask")
def ask(request: AskRequest):
    try:
        return handle_question(
            video_id=request.video_id.strip(),
            question=request.question.strip(),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
