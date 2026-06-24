from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
import pdfplumber

from backend.core.config import get_data_dir
from backend.utils.security import anonymize_identity, normalize_skills, redacted_text, safe_split_skills

LOGGER = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
SECTION_HEADER_RE = re.compile(
    r"^(education|skills|experience|projects|interests|career goal|summary|work experience)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file using PyMuPDF, with a pdfplumber fallback."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    text_parts: list[str] = []
    try:
        with fitz.open(path) as doc:
            for page in doc:
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:  # pragma: no cover - fallback path
        LOGGER.debug("PyMuPDF extraction failed; using pdfplumber fallback: %s", exc)
        text_parts = []

    if not text_parts:
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        text_parts.append(page_text)
        except Exception as exc:
            raise ValueError(f"Unable to extract text from PDF: {exc}") from exc

    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError("The uploaded PDF did not contain extractable text.")
    return text


def _load_skill_taxonomy() -> list[str]:
    taxonomy_path = get_data_dir() / "skill_taxonomy.csv"
    if not taxonomy_path.exists():
        return []
    df = pd.read_csv(taxonomy_path)
    skills = [str(skill) for skill in df.get("skill", pd.Series(dtype=str)).dropna().tolist()]
    extras = [
        "machine learning", "deep learning", "data science", "artificial intelligence",
        "natural language processing", "computer vision", "recommendation systems",
        "model deployment", "model monitoring", "llm", "rag",
    ]
    return normalize_skills(skills + extras)


def parse_skills(text: str) -> list[str]:
    """Extract likely skills from resume text using the local taxonomy and section heuristics."""
    lowered = text.lower()
    skills = _load_skill_taxonomy()
    matches: list[str] = []
    for skill in skills:
        if skill.lower() in lowered:
            matches.append(skill)
    # Additional heuristics for the sample resume and common phrasing.
    heuristic_hits = []
    for phrase, canonical in [
        ("seaborn", "Seaborn"),
        ("matplotlib", "Matplotlib"),
        ("jupyter", "Jupyter Notebook"),
        ("excel", "Excel"),
        ("nltk", "NLTK"),
        ("transformers", "Transformers"),
        ("pytorch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("docker", "Docker"),
        ("fastapi", "FastAPI"),
        ("postgresql", "PostgreSQL"),
        ("kubernetes", "Kubernetes"),
        ("tableau", "Tableau"),
        ("power bi", "Power BI"),
    ]:
        if phrase in lowered:
            heuristic_hits.append(canonical)

    return normalize_skills(matches + heuristic_hits)


def parse_education(text: str) -> dict[str, Any]:
    """Extract basic education details from resume text."""
    edu_match = re.search(
        r"(?im)^(?P<degree>.+?)\s*\|\s*(?P<institution>.+?)\s*\|\s*(?P<years>\d{4}[^\n]*)$",
        text,
    )
    if edu_match:
        return {
            "degree": edu_match.group("degree").strip(),
            "institution": edu_match.group("institution").strip(),
            "years": edu_match.group("years").strip(),
        }

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower() == "education":
            for candidate in lines[idx + 1 : idx + 6]:
                candidate = candidate.strip()
                if candidate and "|" in candidate:
                    parts = [p.strip() for p in candidate.split("|")]
                    if len(parts) >= 3:
                        return {"degree": parts[0], "institution": parts[1], "years": parts[2]}

    return {}


def parse_experience(text: str) -> list[str]:
    """Extract experience and project highlights."""
    sections = ["experience", "work experience", "projects"]
    lines = text.splitlines()
    selected: list[str] = []
    current_section = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        normalized = stripped.lower().rstrip(":")
        if normalized in sections:
            current_section = normalized
            continue
        if current_section in sections:
            if stripped.isupper() and len(stripped.split()) <= 5:
                current_section = None
                continue
            if not SECTION_HEADER_RE.match(stripped):
                if len(stripped) > 2:
                    selected.append(stripped)
    return selected[:12]


def build_profile(text: str, user_inputs: dict) -> dict:
    """Combine user inputs and parsed resume data into a structured profile."""
    resume_text = redacted_text(text)
    parsed_skills = parse_skills(resume_text)
    education = parse_education(resume_text)
    experience = parse_experience(resume_text)
    user_skills = safe_split_skills(user_inputs.get("skills"))
    interests = safe_split_skills(user_inputs.get("interests"))
    merged_skills = normalize_skills(user_skills + parsed_skills)

    name = (user_inputs.get("name") or "").strip()
    career_goal = (user_inputs.get("career_goal") or "").strip()
    profile = {
        "user_id": anonymize_identity(name or None, career_goal or None),
        "display_name": name,
        "career_goal": career_goal,
        "skills": merged_skills,
        "user_supplied_skills": user_skills,
        "parsed_resume_skills": parsed_skills,
        "interests": interests,
        "education": education,
        "experience": experience,
        "resume_summary": _summarize_resume_text(resume_text),
    }
    return profile


def _summarize_resume_text(text: str, max_chars: int = 900) -> str:
    text = redacted_text(text)
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:max_chars]
