from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.utils.security import normalize_skill
from recommendation_engine.skill_gap_analysis import identify_skill_gaps


def _safe_cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    a_vec = np.asarray(a, dtype=np.float32)
    b_vec = np.asarray(b, dtype=np.float32)
    if a_vec.size == 0 or b_vec.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def _parse_skill_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[,;|\n]", str(value))
    cleaned = [item.strip() for item in raw if str(item).strip()]
    seen = set()
    result = []
    for item in cleaned:
        norm = normalize_skill(item)
        if norm not in seen:
            seen.add(norm)
            result.append(item)
    return result


def calculate_skill_match(student_skills: list[str], required_skills: list[str]) -> float:
    """Returns fraction of required skills the student already has (0.0 – 1.0)."""
    if not required_skills:
        return 0.0
    student_set = {normalize_skill(s) for s in student_skills if s}
    required_norm = [normalize_skill(s) for s in required_skills if s]
    if not required_norm:
        return 0.0
    matched = sum(1 for skill in required_norm if skill in student_set)
    return matched / len(required_norm)


def calculate_domain_match(profile_embedding: list[float], record_embedding: list[float]) -> float:
    """Returns cosine similarity between two embeddings (0.0 – 1.0)."""
    similarity = _safe_cosine(profile_embedding, record_embedding)
    return max(0.0, min(1.0, similarity))


def calculate_goal_alignment(career_goal: str, role_description: str) -> float:
    """Returns text similarity between career goal and role description (0.0 – 1.0)."""
    texts = [career_goal or "", role_description or ""]
    if not any(texts):
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        return 0.0
    sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return max(0.0, min(1.0, float(sim)))


def calculate_resource_availability(skill_gaps: list[str], resources_df: pd.DataFrame | None) -> float:
    """Returns fraction of skill gaps that have learning resources available (0.0 – 1.0)."""
    if not skill_gaps:
        return 1.0
    if resources_df is None or resources_df.empty:
        return 0.0

    searchable_cols = [c for c in ["title", "topic", "type", "skills_covered", "level"] if c in resources_df.columns]
    if not searchable_cols:
        return 0.0

    covered = 0
    for gap in skill_gaps:
        gap_norm = normalize_skill(gap)
        mask = pd.Series(False, index=resources_df.index)
        for col in searchable_cols:
            mask = mask | resources_df[col].astype(str).str.lower().str.contains(re.escape(gap_norm), na=False)
        if mask.any():
            covered += 1
    return covered / len(skill_gaps)


def calculate_confidence_score(
    skill_match: float,
    domain_match: float,
    goal_alignment: float,
    resource_availability: float,
) -> float:
    """Final confidence score (0.0 – 1.0)."""
    return (
        skill_match * 0.40
        + domain_match * 0.25
        + goal_alignment * 0.20
        + resource_availability * 0.15
    )


def rank_results(results: list[dict[str, Any]], profile: dict, resources_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Score and sort all retrieved results by confidence score (descending)."""
    ranked: list[dict[str, Any]] = []
    student_skills = profile.get("skills", [])
    profile_embedding = profile.get("embedding", [])
    career_goal = profile.get("career_goal", "")

    for item in results:
        payload = item.get("payload", item)
        required_skills = _parse_skill_text(payload.get("skills_required") or payload.get("required_skills"))
        record_embedding = item.get("embedding", [])
        record_text = item.get("text", "")
        description = payload.get("description") or record_text

        skill_match = calculate_skill_match(student_skills, required_skills)
        domain_match = calculate_domain_match(profile_embedding, record_embedding)
        goal_alignment = calculate_goal_alignment(career_goal, description)

        gaps = identify_skill_gaps(student_skills, required_skills)
        resource_availability = calculate_resource_availability(gaps, resources_df)
        confidence = calculate_confidence_score(skill_match, domain_match, goal_alignment, resource_availability)

        reasons = [
            f"Skill match: {skill_match * 100:.0f}% of required skills already present",
            f"Domain similarity: {domain_match:.2f} cosine match against profile embedding",
            f"Goal alignment: {goal_alignment:.2f} text similarity with career goal",
            f"Resource coverage: {resource_availability * 100:.0f}% of missing skills have learning resources",
        ]
        enriched = dict(payload)
        enriched.update(
            {
                "id": item.get("id"),
                "collection": item.get("collection"),
                "similarity": float(item.get("similarity", 0.0)),
                "required_skills": required_skills,
                "skill_gaps": gaps,
                "confidence_score": round(confidence * 100, 2),
                "confidence_score_raw": round(confidence, 4),
                "reasons": reasons,
                "domain_match": round(domain_match, 4),
                "goal_alignment": round(goal_alignment, 4),
                "skill_match": round(skill_match, 4),
                "resource_availability": round(resource_availability, 4),
            }
        )
        ranked.append(enriched)

    ranked.sort(key=lambda row: row["confidence_score_raw"], reverse=True)
    return ranked
