from __future__ import annotations

from pathlib import Path

import pandas as pd


def _record_to_document(row: pd.Series, id_prefix: str, text_fields: list[str], metadata_fields: list[str]) -> dict:
    text_parts = []
    for field in text_fields:
        if field in row and pd.notna(row[field]):
            text_parts.append(f"{field}: {row[field]}")
    metadata = {field: row[field] for field in metadata_fields if field in row and pd.notna(row[field])}
    return {
        "id": f"{id_prefix}_{int(row.name)}",
        "text": "\n".join(text_parts),
        "metadata": metadata,
    }


def load_internship_documents(csv_path: str) -> list[dict]:
    """Load internships CSV and convert each row into a document dict."""
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        docs.append(
            _record_to_document(
                row,
                "internship",
                ["title", "company", "location", "skills_required", "domain", "stipend", "duration", "description"],
                ["title", "company", "location", "skills_required", "domain", "stipend", "duration"],
            )
        )
    return docs


def load_career_documents(csv_path: str) -> list[dict]:
    """Load careers CSV into document format."""
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        docs.append(
            _record_to_document(
                row,
                "career",
                ["role_title", "description", "required_skills", "average_salary", "growth_rate", "related_roles"],
                ["role_title", "description", "required_skills", "average_salary", "growth_rate", "related_roles"],
            )
        )
    return docs


def load_resource_documents(csv_path: str) -> list[dict]:
    """Load learning resources CSV into document format."""
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        docs.append(
            _record_to_document(
                row,
                "resource",
                ["title", "type", "topic", "url", "level", "skills_covered"],
                ["title", "type", "topic", "url", "level", "skills_covered"],
            )
        )
    return docs
