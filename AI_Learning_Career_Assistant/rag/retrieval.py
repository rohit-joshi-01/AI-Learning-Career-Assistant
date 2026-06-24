from __future__ import annotations

import json
import os
from typing import Any

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except Exception:  # pragma: no cover - fallback
    genai = None

from backend.core.config import get_settings


def retrieve_context(query_embedding: list[float], collection, top_k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top_k relevant documents from a loaded collection or local store."""
    if collection is None:
        return []
    try:
        query = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        documents = query.get("documents", [[]])[0]
        metadatas = query.get("metadatas", [[]])[0]
        ids = query.get("ids", [[]])[0]
        return [
            {
                "id": ids[idx] if idx < len(ids) else None,
                "text": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
            }
            for idx in range(min(len(documents), top_k))
        ]
    except Exception:
        return []


def _fallback_payload(profile: dict, recommendation: dict) -> dict:
    role = recommendation.get("role_title") or recommendation.get("title") or "this role"
    return {
        "explanation": (
            f"{role} matches your current strengths in {', '.join(profile.get('skills', [])[:4]) or 'your profile'}. "
            f"The role also aligns with your goal of {profile.get('career_goal', 'career growth')}. "
            f"Recommended confidence: {recommendation.get('confidence_score', 0)}%."
        ),
        "learning_roadmap": [
            "Month 1: Strengthen the missing core skills with one beginner course per gap.",
            "Month 2: Build 1–2 portfolio projects that demonstrate the target role skills.",
            "Month 3: Practice role-specific interview questions and refine your resume.",
        ],
        "interview_tips": [
            "Prepare one project story using the STAR method.",
            "Review fundamentals behind the skills listed in the recommendation.",
            "Practice explaining how you would deploy or evaluate a solution.",
        ],
        "profile_summary": (
            f"Profile summary for anonymous user {profile.get('user_id', 'unknown')}: "
            f"{len(profile.get('skills', []))} skills, {len(profile.get('interests', []))} interests."
        ),
        "disclaimer": "These suggestions are AI-generated. Verify before acting on them.",
    }


def generate_explanation(profile: dict, recommendation: dict, context_docs: list[dict]) -> dict:
    """
    Use Gemini API to explain WHY this recommendation was made.
    LLM role: explanation ONLY — not selection.
    """
    api_key = get_settings()["gemini_api_key"]
    if genai is None or not api_key:
        return _fallback_payload(profile, recommendation)

    try:  # pragma: no cover - depends on external service
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = {
            "role": "system",
            "content": (
                "You are a career guidance assistant. "
                "You must only explain already-selected recommendations, summarize the profile, "
                "generate a learning roadmap, and provide interview tips. "
                "Respond in valid JSON with keys: explanation, learning_roadmap, interview_tips, profile_summary, disclaimer."
            ),
        }
        user_prompt = {
            "profile": {
                "skills": profile.get("skills", []),
                "interests": profile.get("interests", []),
                "career_goal": profile.get("career_goal", ""),
                "education": profile.get("education", {}),
            },
            "recommendation": {
                "title": recommendation.get("role_title") or recommendation.get("title"),
                "confidence_score": recommendation.get("confidence_score"),
                "skill_gaps": recommendation.get("skill_gaps", []),
                "reasons": recommendation.get("reasons", []),
            },
            "context_documents": context_docs[:5],
        }
        response = model.generate_content(
            json.dumps({"prompt": prompt, "payload": user_prompt}, ensure_ascii=False),
        )
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return _fallback_payload(profile, recommendation)
    except Exception:
        return _fallback_payload(profile, recommendation)
