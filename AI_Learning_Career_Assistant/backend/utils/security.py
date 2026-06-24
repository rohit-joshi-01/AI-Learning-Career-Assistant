from __future__ import annotations

import hashlib
import re
from typing import Iterable


EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
NAME_RE = re.compile(r"\bName\s*:\s*(.+)", re.IGNORECASE)


def redacted_text(text: str) -> str:
    """Redact common PII patterns before logging or displaying debug traces."""
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def anonymize_identity(name: str | None, career_goal: str | None = None) -> str:
    """Create a stable anonymous identifier for transient sessions."""
    raw = f"{name or ''}|{career_goal or ''}".strip("|")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"user_{digest[:12]}"


def normalize_skill(skill: str) -> str:
    return re.sub(r"\s+", " ", skill.strip().lower())


def normalize_skills(skills: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for skill in skills:
        value = normalize_skill(skill)
        if value and value not in seen:
            seen.add(value)
            normalized.append(skill.strip())
    return normalized


def safe_split_skills(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return normalize_skills(value)
    parts = [p.strip() for p in re.split(r"[,;|\n]", value) if p.strip()]
    return normalize_skills(parts)


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(p.lower() in lowered for p in phrases)
