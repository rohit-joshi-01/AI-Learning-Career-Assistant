from __future__ import annotations

from typing import Any

from rag.retrieval import generate_explanation


def build_llm_output(profile: dict, recommendation: dict, context_docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate explanation, roadmap, interview tips, and profile summary."""
    payload = generate_explanation(profile, recommendation, context_docs)
    payload.setdefault("learning_roadmap", [])
    payload.setdefault("interview_tips", [])
    payload.setdefault("profile_summary", "")
    payload.setdefault("explanation", "")
    payload.setdefault("disclaimer", "These suggestions are AI-generated. Verify before acting on them.")
    return payload
