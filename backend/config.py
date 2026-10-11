"""Application configuration: .env defaults plus admin-editable runtime overrides."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def _resolve_path(value: str, base_dir: Path = BASE_DIR) -> str:
    """Resolve relative paths from the backend directory."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


# Settings the admin dashboard may change at runtime: name -> (type, is_secret).
EDITABLE_SETTINGS: Dict[str, tuple] = {
    "LLM_PROVIDER": (str, False),
    "LLM_MODEL": (str, False),
    "OPENAI_API_KEY": (str, True),
    "OPENAI_BASE_URL": (str, False),
    "GEMINI_API_KEY": (str, True),
    "GEMINI_API_ENDPOINT": (str, False),
    "EMBEDDING_MODEL": (str, False),
    "CHUNK_SIZE": (int, False),
    "CHUNK_OVERLAP": (int, False),
    "TOP_K": (int, False),
    "TEMPERATURE": (float, False),
    "MAX_TOKENS": (int, False),
    "SITE_URL": (str, False),
    "RATE_LIMIT_PER_MINUTE": (int, False),
}

# Changing any of these invalidates the vector index.
INDEX_SETTINGS = {"EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP"}
LLM_SETTINGS = {"LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", "OPENAI_BASE_URL", "GEMINI_API_KEY", "GEMINI_API_ENDPOINT"}


class Settings:
    """Application settings from environment variables, overridable from the admin dashboard."""

    def __init__(self) -> None:
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.tapsage.com/openai/v1")
        self.GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        self.GEMINI_API_ENDPOINT: str = os.getenv("GEMINI_API_ENDPOINT", "https://api.tapsage.com")
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")
        self.LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
        self.TOP_K: int = int(os.getenv("TOP_K", "6"))
        self.TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.5"))
        self.MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
        self.SITE_URL: str = os.getenv("SITE_URL", "https://tabenergy.ir")
        self.RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
        self.ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")
        self.CHROMA_DB_PATH: str = _resolve_path(os.getenv("CHROMA_DB_PATH", "../chroma_db"))
        self.PRODUCT_DB_PATH: str = _resolve_path(os.getenv("PRODUCT_DB_PATH", "products.db"))
        self.KNOWLEDGE_BASE_DIR: str = _resolve_path(os.getenv("KNOWLEDGE_BASE_DIR", "../knowledge_base"))
        self.RAW_DOCS_DIR: str = str(Path(self.KNOWLEDGE_BASE_DIR) / "raw_docs")
        self.DATA_DIR: str = _resolve_path(os.getenv("DATA_DIR", "data"))
        self.CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "farsi_rag_collection")
        self.CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:8080").split(",")
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))
        self._load_overrides()

    @property
    def overrides_path(self) -> Path:
        return Path(self.DATA_DIR) / "runtime_settings.json"

    def editable_values(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in EDITABLE_SETTINGS}

    def update(self, changes: Dict[str, Any]) -> None:
        """Apply already-validated changes and persist them."""
        for name, value in changes.items():
            setattr(self, name, value)
        path = self.overrides_path
        path.parent.mkdir(parents=True, exist_ok=True)
        current = self._read_overrides()
        current.update(changes)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _read_overrides(self) -> Dict[str, Any]:
        path = self.overrides_path
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Ignoring unreadable runtime settings %s: %s", path, exc)
            return {}

    def _load_overrides(self) -> None:
        for name, value in self._read_overrides().items():
            if name in EDITABLE_SETTINGS:
                cast, _ = EDITABLE_SETTINGS[name]
                setattr(self, name, cast(value))


settings = Settings()
