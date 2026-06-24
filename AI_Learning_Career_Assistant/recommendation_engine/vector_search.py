from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from backend.services.storage import load_records, records_exist
from recommendation_engine.embedding_generator import generate_embedding

LOGGER = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_vec = np.asarray(a, dtype=np.float32)
    b_vec = np.asarray(b, dtype=np.float32)
    if a_vec.size == 0 or b_vec.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def _query_local_store(collection_name: str, profile_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    records = load_records(collection_name)
    scored: list[dict[str, Any]] = []
    for record in records:
        emb = record.get("embedding") or []
        sim = _cosine_similarity(profile_embedding, emb)
        item = dict(record)
        item["similarity"] = sim
        scored.append(item)
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def search_internships(profile_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """Return top_k most similar internship records."""
    return _query_local_store("internships", profile_embedding, top_k=top_k)


def search_careers(profile_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """Return top_k most similar career records."""
    return _query_local_store("careers", profile_embedding, top_k=top_k)


def search_resources(profile_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """Return top_k most similar learning resource records."""
    return _query_local_store("learning_resources", profile_embedding, top_k=top_k)


def search_collection(collection_name: str, profile_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    return _query_local_store(collection_name, profile_embedding, top_k=top_k)
