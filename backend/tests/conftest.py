"""Test setup: isolated temp data dirs, fixed settings, and a fake LLM + index (no network, no model download)."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# Must be set before config is imported; load_dotenv() never overrides existing variables.
_TMP = Path(tempfile.mkdtemp(prefix="chatbot-tests-"))
os.environ.update(
    {
        "ADMIN_PASSWORD": "test-password",
        "DATA_DIR": str(_TMP / "data"),
        "PRODUCT_DB_PATH": str(_TMP / "products.db"),
        "CHROMA_DB_PATH": str(_TMP / "chroma"),
        "KNOWLEDGE_BASE_DIR": str(_TMP / "knowledge_base"),
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "test-model",
        "OPENAI_API_KEY": "sk-test-key-1234",
        "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
        "GEMINI_API_KEY": "",
        "GEMINI_API_ENDPOINT": "",
        "EMBEDDING_MODEL": "test-embedding",
        "CHUNK_SIZE": "800",
        "CHUNK_OVERLAP": "150",
        "TOP_K": "6",
        "TEMPERATURE": "0.5",
        "MAX_TOKENS": "1000",
        "SITE_URL": "https://tabenergy.ir",
        "RATE_LIMIT_PER_MINUTE": "1000",
    }
)

from config import EDITABLE_SETTINGS, settings  # noqa: E402
from links import LinkResolver  # noqa: E402

CONTACT_URL = "https://tabenergy.ir/contact/"
SERVICE_URL = "https://tabenergy.ir/%d8%ae%d8%af%d9%85%d8%a7%d8%aa/774/"  # tabenergy.ir/خدمات/774/


class FakeLLM:
    """Answers the analysis prompt with JSON and streams a reply containing a hand-typed Persian URL."""

    provider = "fake"

    def __init__(self):
        self.stream_calls = 0

    def chat(self, messages, temperature=0.0, max_tokens=None):
        question = messages[-1]["content"]
        if "JSON" in messages[0]["content"]:
            intent = "greeting" if '"سلام"' in question else "general_support"
            return json.dumps({"intent": intent, "standalone_query": "تماس", "corrected_query": "تماس", "product_keywords": []})
        return "".join(self.stream_chat(messages, temperature, max_tokens))

    def stream_chat(self, messages, temperature=0.0, max_tokens=None):
        self.stream_calls += 1
        yield "برای خدمات به "
        yield "tabenergy.ir/خدمات/774 "
        yield "مراجعه کنید."


class FakeIndex:
    ready = True
    stats = {"sources": 1, "chunks": 1, "sources_by_type": {"page": 1}}

    def __init__(self):
        self.resolver = LinkResolver([CONTACT_URL, SERVICE_URL], site_url="https://tabenergy.ir")
        self.builds = 0

    def search(self, queries, top_k):
        return [("تلفن: ۰۱۱-۴۲۴۳-۳۴۴۳", {"source_title": "تماس با ما", "source_url": CONTACT_URL, "source_type": "page", "categories": ""})]

    def build(self, force=False, progress=lambda message: None):
        self.builds += 1
        return {**self.stats, "chunk_size": settings.CHUNK_SIZE}


@pytest.fixture(autouse=True)
def restore_settings():
    """Undo setting changes and saved overrides after each test."""
    saved = {name: getattr(settings, name) for name in EDITABLE_SETTINGS}
    yield
    for name, value in saved.items():
        setattr(settings, name, value)
    settings.overrides_path.unlink(missing_ok=True)


@pytest.fixture
def app_env():
    """The FastAPI app with the fake LLM and index. Lifespan is not run, so no website sync starts."""
    import main
    import security

    security._limiter = security.RateLimiter()
    fake_llm, fake_index = FakeLLM(), FakeIndex()
    main.chatbot.llm, main.chatbot.llm_error = fake_llm, ""
    main.chatbot.index = fake_index
    main.chatbot.reload_llm = lambda: None  # keep the fake when LLM settings change
    from fastapi.testclient import TestClient

    return TestClient(main.app), fake_llm, fake_index


@pytest.fixture
def client(app_env):
    return app_env[0]


@pytest.fixture
def admin_headers(client):
    token = client.post("/admin/login", json={"password": "test-password"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}
