from __future__ import annotations

from backend.utils.security import normalize_skill


def identify_skill_gaps(student_skills: list[str], required_skills: list[str]) -> list[str]:
    """Return list of skills the student is missing."""
    student_set = {normalize_skill(s) for s in student_skills if s}
    gaps = []
    for required in required_skills:
        normalized = normalize_skill(required)
        if normalized and normalized not in student_set:
            gaps.append(required.strip())
    return gaps


def format_skill_gap_report(role_title: str, gaps: list[str]) -> dict:
    """Return a structured skill gap report for one role."""
    return {
        "role": role_title,
        "missing_skills": gaps,
        "gap_count": len(gaps),
    }
