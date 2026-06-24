from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback
    SentenceTransformer = None

from sklearn.feature_extraction.text import HashingVectorizer

MODEL_NAME = "all-MiniLM-L6-v2"
_HASHING_DIMENSIONS = 384
_model = None
_hashing_vectorizer = HashingVectorizer(n_features=_HASHING_DIMENSIONS, alternate_sign=False, norm="l2")


def _fallback_encode(texts: list[str]) -> np.ndarray:
    matrix = _hashing_vectorizer.transform(texts)
    return matrix.toarray().astype(np.float32)


def get_model():
    global _model
    if _model is None and SentenceTransformer is not None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def generate_embedding(text: str) -> list[float]:
    """Generate a single embedding vector for the given text."""
    if not text:
        text = " "
    model = get_model()
    if model is not None:
        embedding = model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()
    return _fallback_encode([text])[0].tolist()


def generate_batch_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts (batch processing)."""
    if not texts:
        return []
    model = get_model()
    if model is not None:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    return _fallback_encode(texts).tolist()


def embedding_dim() -> int:
    return _HASHING_DIMENSIONS if get_model() is None else 384
