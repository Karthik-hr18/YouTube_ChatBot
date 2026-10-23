from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Server.settings import FRONTEND_ORIGINS
from Server.controllers import router


app = FastAPI()
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
