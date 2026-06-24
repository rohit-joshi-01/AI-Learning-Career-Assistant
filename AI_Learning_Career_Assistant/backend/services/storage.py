from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.core.config import get_store_dir

LOGGER = logging.getLogger(__name__)

SUPPORTED_COLLECTIONS = ("internships", "careers", "learning_resources")


@dataclass(slots=True)
class LocalRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]
    payload: dict[str, Any]


def _collection_path(collection_name: str) -> Path:
    if collection_name not in SUPPORTED_COLLECTIONS:
        raise ValueError(f"Unsupported collection: {collection_name}")
    return get_store_dir() / f"{collection_name}_store.json"


def save_records(collection_name: str, records: list[dict[str, Any]]) -> None:
    path = _collection_path(collection_name)
    serializable = []
    for record in records:
        embedding = record.get("embedding", [])
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        serializable.append(
            {
                "id": str(record.get("id")),
                "text": record.get("text", ""),
                "embedding": embedding,
                "metadata": record.get("metadata", {}),
                "payload": record.get("payload", {}),
            }
        )
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s records to %s", len(serializable), path)


def load_records(collection_name: str) -> list[dict[str, Any]]:
    path = _collection_path(collection_name)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def records_exist(collection_name: str) -> bool:
    return _collection_path(collection_name).exists()


def ensure_manifest() -> None:
    manifest = {
        "backend": "local-json-fallback",
        "collections": SUPPORTED_COLLECTIONS,
    }
    (get_store_dir() / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
