from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from backend.core.config import get_data_dir, get_store_dir
from backend.services.storage import ensure_manifest, save_records
from recommendation_engine.embedding_generator import generate_batch_embeddings
from rag.document_loader import load_career_documents, load_internship_documents, load_resource_documents

LOGGER = logging.getLogger(__name__)


def _build_records(dataset_name: str, docs: list[dict[str, Any]], payload_key: str) -> list[dict[str, Any]]:
    texts = [doc["text"] for doc in docs]
    embeddings = generate_batch_embeddings(texts)
    records = []
    for doc, embedding in zip(docs, embeddings, strict=True):
        records.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "embedding": embedding,
                "metadata": doc["metadata"],
                "payload": doc["metadata"] | {payload_key: doc["metadata"].get(payload_key)} if payload_key in doc["metadata"] else doc["metadata"],
            }
        )
    return records


def load_internships() -> None:
    """Load internships.csv into local vector storage."""
    csv_path = get_data_dir() / "internships.csv"
    docs = load_internship_documents(str(csv_path))
    records = _build_records("internships", docs, "title")
    save_records("internships", records)


def load_careers() -> None:
    """Load careers.csv into local vector storage."""
    csv_path = get_data_dir() / "careers.csv"
    docs = load_career_documents(str(csv_path))
    records = _build_records("careers", docs, "role_title")
    save_records("careers", records)


def load_learning_resources() -> None:
    """Load learning_resources.csv into local vector storage."""
    csv_path = get_data_dir() / "learning_resources.csv"
    docs = load_resource_documents(str(csv_path))
    records = _build_records("learning_resources", docs, "title")
    save_records("learning_resources", records)


def build_all_indexes(force: bool = False) -> dict[str, int]:
    """Build all local indexes. Returns record counts."""
    ensure_manifest()
    counts: dict[str, int] = {}

    loaders = {
        "internships": load_internships,
        "careers": load_careers,
        "learning_resources": load_learning_resources,
    }
    for name, loader in loaders.items():
        store_path = get_store_dir() / f"{name}_store.json"
        if force or not store_path.exists():
            loader()
        records = json.loads(store_path.read_text(encoding="utf-8"))
        counts[name] = len(records)
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    counts = build_all_indexes(force=True)
    print("Loaded datasets into vector storage:")
    for key, value in counts.items():
        print(f"  - {key}: {value} records")
