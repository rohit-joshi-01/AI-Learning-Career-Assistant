from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR: Final[Path] = PROJECT_ROOT / "datasets"
DEFAULT_STORE_DIR: Final[Path] = PROJECT_ROOT / "chroma_db"


@lru_cache(maxsize=1)
def get_settings() -> dict[str, str]:
    """Return environment-backed configuration values."""
    data_dir = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser().resolve()
    store_dir = Path(os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_STORE_DIR))).expanduser().resolve()
    return {
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "data_dir": str(data_dir),
        "store_dir": str(store_dir),
        "app_log_level": os.getenv("APP_LOG_LEVEL", "INFO").upper(),
        "backend_host": os.getenv("BACKEND_HOST", "127.0.0.1"),
        "backend_port": os.getenv("BACKEND_PORT", "8000"),
        "streamlit_backend_url": os.getenv("STREAMLIT_BACKEND_URL", "http://127.0.0.1:8000"),
    }


def get_data_dir() -> Path:
    return Path(get_settings()["data_dir"])


def get_store_dir() -> Path:
    path = Path(get_settings()["store_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path
