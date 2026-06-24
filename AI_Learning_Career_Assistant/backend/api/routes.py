from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.schemas import RecommendationResponse
from backend.services.pipeline import RecommendationPipeline
from backend.services.storage import load_records

router = APIRouter()
_pipeline = RecommendationPipeline()


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(
    resume: UploadFile = File(...),
    name: str = Form(""),
    career_goal: str = Form(""),
    skills: str = Form(""),
    interests: str = Form(""),
) -> dict[str, Any]:
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file.")

    file_bytes = await resume.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume is empty.")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    user_inputs = {
        "name": name,
        "career_goal": career_goal,
        "skills": [s.strip() for s in skills.split(",") if s.strip()],
        "interests": [s.strip() for s in interests.split(",") if s.strip()],
    }
    result = _pipeline.recommend_from_upload(user_inputs, str(tmp_path), top_k=5)
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass
    return result


@router.post("/skill-gap")
async def skill_gap(
    resume: UploadFile = File(...),
    name: str = Form(""),
    career_goal: str = Form(""),
    skills: str = Form(""),
    interests: str = Form(""),
) -> dict[str, Any]:
    result = await recommend(
        resume=resume,
        name=name,
        career_goal=career_goal,
        skills=skills,
        interests=interests,
    )
    return {"user_id": result["user_id"], "skill_gap_summary": result["skill_gap_summary"]}


@router.get("/internships")
def internships(limit: int = 20) -> dict[str, Any]:
    items = load_records("internships")[:limit]
    return {"count": len(items), "items": items}


@router.get("/careers")
def careers(limit: int = 20) -> dict[str, Any]:
    items = load_records("careers")[:limit]
    return {"count": len(items), "items": items}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "AI Career Assistant API is running"}
