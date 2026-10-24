import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Server.settings import FRONTEND_ORIGINS
from Server.controllers import job_worker, router
from Server.job_store import initialize_job_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_job_store()
    worker_task = asyncio.create_task(job_worker())
    app.state.job_worker_task = worker_task
    try:
        yield
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(FRONTEND_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
