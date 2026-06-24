from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.services.data_loader import build_all_indexes

app = FastAPI(
    title="AI Career Assistant API",
    description="Backend for the AI Learning & Career Assistant internship project",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def _startup() -> None:
    build_all_indexes(force=False)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Career Assistant API is running"}


# Run: uvicorn backend.app:app --reload
