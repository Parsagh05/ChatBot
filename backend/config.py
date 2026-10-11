"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def _resolve_path(value: str, base_dir: Path = BASE_DIR) -> str:
    """Resolve relative paths from the backend directory."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


class Settings:
    """Application settings from environment variables."""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.tapsage.com/openai/v1")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    GEMINI_API_ENDPOINT: str = os.getenv("GEMINI_API_ENDPOINT", "https://api.tapsage.com")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    CHROMA_DB_PATH: str = _resolve_path(os.getenv("CHROMA_DB_PATH", "../chroma_db"))
    PRODUCT_DB_PATH: str = _resolve_path(os.getenv("PRODUCT_DB_PATH", "products.db"))
    KNOWLEDGE_BASE_DIR: str = _resolve_path(os.getenv("KNOWLEDGE_BASE_DIR", "../knowledge_base"))
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "farsi_rag_collection")
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:8080").split(",")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()

