"""Admin dashboard API: runtime settings, knowledge-base jobs and system status."""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from config import EDITABLE_SETTINGS, INDEX_SETTINGS, LLM_SETTINGS, settings
from product_db import count_products, import_products, list_feedback
from security import login, logout, require_admin
from site_crawler import crawl_site, load_snapshot

logger = logging.getLogger(__name__)

NUMBER_LIMITS = {
    "CHUNK_SIZE": (200, 4000),
    "CHUNK_OVERLAP": (0, 1000),
    "TOP_K": (1, 20),
    "TEMPERATURE": (0.0, 2.0),
    "MAX_TOKENS": (64, 8192),
    "RATE_LIMIT_PER_MINUTE": (1, 1000),
}
CHOICES = {"LLM_PROVIDER": {"auto", "openai", "gemini"}}


class JobRunner:
    """Runs one background job at a time and records its progress."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state: Dict[str, Any] = {"name": None, "status": "idle", "message": "", "started_at": None, "finished_at": None}

    @property
    def running(self) -> bool:
        return self.state["status"] == "running"

    def start(self, name: str, work: Callable[[Callable[[str], None]], str]) -> bool:
        with self._lock:
            if self.running:
                return False
            self.state = {"name": name, "status": "running", "message": "", "started_at": _now(), "finished_at": None}
        threading.Thread(target=self._run, args=(work,), daemon=True, name=f"job-{name}").start()
        return True

    def _run(self, work: Callable[[Callable[[str], None]], str]) -> None:
        def progress(message: str) -> None:
            self.state["message"] = message

        try:
            message = work(progress)
            self.state.update(status="done", message=message)
        except Exception as exc:
            logger.exception("Job %s failed", self.state["name"])
            self.state.update(status="error", message=str(exc))
        finally:
            self.state["finished_at"] = _now()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


jobs = JobRunner()
router = APIRouter(prefix="/admin", tags=["admin"])
_chatbot = None  # set by bind()


def bind(chatbot) -> None:
    global _chatbot
    _chatbot = chatbot


# ─── Jobs ────────────────────────────────────────────────────────────────────


def _sync_site(progress: Callable[[str], None]) -> str:
    snapshot = crawl_site(progress)
    progress("در حال به‌روزرسانی پایگاه داده محصولات...")
    product_count = import_products(snapshot["products"])
    stats = _chatbot.index.build(progress=progress)
    return f"همگام‌سازی کامل شد: {len(snapshot['documents'])} صفحه/مطلب، {product_count} محصول، {stats['chunks']} بخش در ایندکس."


def _rebuild_index(force: bool) -> Callable[[Callable[[str], None]], str]:
    def work(progress: Callable[[str], None]) -> str:
        stats = _chatbot.index.build(force=force, progress=progress)
        return f"ایندکس آماده است: {stats['sources']} منبع، {stats['chunks']} بخش."

    return work


def start_startup_job() -> None:
    """Load the index on startup; fetch the website first if it has never been synced."""
    snapshot = load_snapshot()
    if snapshot is None:
        jobs.start("sync", _sync_site)
        return

    def work(progress: Callable[[str], None]) -> str:
        import_products(snapshot.get("products", []))
        return _rebuild_index(force=False)(progress)

    jobs.start("startup", work)


# ─── Endpoints ───────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    password: str


@router.post("/login")
def admin_login(body: LoginRequest, request: Request):
    return {"token": login(request, body.password)}


@router.post("/logout")
def admin_logout(token: str = Depends(require_admin)):
    logout(token)
    return {"status": "ok"}


@router.get("/settings", dependencies=[Depends(require_admin)])
def get_settings():
    values: Dict[str, Any] = {}
    for name, (_, secret) in EDITABLE_SETTINGS.items():
        value = getattr(settings, name)
        values[name] = {"set": bool(value), "hint": f"…{value[-4:]}" if len(value) > 8 else ""} if secret else value
    return {"settings": values}


@router.put("/settings", dependencies=[Depends(require_admin)])
def update_settings(changes: Dict[str, Any]):
    validated = _validate(changes)
    changed = {name: value for name, value in validated.items() if getattr(settings, name) != value}
    if not changed:
        return {"changed": [], "message": "تغییری ثبت نشد."}

    settings.update(changed)
    result: Dict[str, Any] = {"changed": sorted(changed), "llm_error": None, "reindex": None}

    if LLM_SETTINGS & changed.keys():
        _chatbot.reload_llm()
        result["llm_error"] = _chatbot.llm_error or None
    if INDEX_SETTINGS & changed.keys():
        result["reindex"] = "started" if jobs.start("reindex", _rebuild_index(force=False)) else "busy"
    return result


def _validate(changes: Dict[str, Any]) -> Dict[str, Any]:
    validated: Dict[str, Any] = {}
    for name, value in changes.items():
        if name not in EDITABLE_SETTINGS:
            raise HTTPException(status_code=400, detail=f"Unknown setting: {name}")
        cast, secret = EDITABLE_SETTINGS[name]
        if secret and (value is None or value == ""):
            continue  # blank secret field means "keep the current key"
        try:
            value = cast(value.strip() if isinstance(value, str) else value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{name} must be a {cast.__name__}")
        if name in NUMBER_LIMITS:
            low, high = NUMBER_LIMITS[name]
            if not low <= value <= high:
                raise HTTPException(status_code=400, detail=f"{name} must be between {low} and {high}")
        if name in CHOICES and value not in CHOICES[name]:
            raise HTTPException(status_code=400, detail=f"{name} must be one of {sorted(CHOICES[name])}")
        if cast is str and not secret and not value and name not in ("OPENAI_BASE_URL", "GEMINI_API_ENDPOINT"):
            raise HTTPException(status_code=400, detail=f"{name} cannot be empty")
        validated[name] = value

    size = validated.get("CHUNK_SIZE", settings.CHUNK_SIZE)
    overlap = validated.get("CHUNK_OVERLAP", settings.CHUNK_OVERLAP)
    if overlap >= size // 2:
        raise HTTPException(status_code=400, detail="CHUNK_OVERLAP must be less than half of CHUNK_SIZE")
    return validated


@router.get("/status", dependencies=[Depends(require_admin)])
def status():
    index = _chatbot.index
    return {
        "llm": {
            "provider": _chatbot.llm.provider if _chatbot.llm else None,
            "model": settings.LLM_MODEL,
            "ready": _chatbot.llm is not None,
            "error": _chatbot.llm_error or None,
        },
        "index": {"ready": index.ready, **index.stats},
        "products": count_products(),
        "job": jobs.state,
    }


@router.get("/feedback", dependencies=[Depends(require_admin)])
def feedback(
    rating: Optional[Literal["up", "down"]] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """User ratings of chatbot answers, newest first."""
    return list_feedback(rating=rating, limit=limit, offset=offset)


@router.post("/sync", dependencies=[Depends(require_admin)])
def sync_site():
    if not jobs.start("sync", _sync_site):
        raise HTTPException(status_code=409, detail="A job is already running.")
    return {"status": "started"}


@router.post("/reindex", dependencies=[Depends(require_admin)])
def reindex():
    if not jobs.start("reindex", _rebuild_index(force=True)):
        raise HTTPException(status_code=409, detail="A job is already running.")
    return {"status": "started"}


@router.post("/test-llm", dependencies=[Depends(require_admin)])
def test_llm():
    if not _chatbot.llm:
        return {"ok": False, "error": _chatbot.llm_error or "LLM is not configured."}
    started = time.perf_counter()
    try:
        reply = _chatbot.llm.chat([{"role": "user", "content": "Reply with the single word: OK"}], temperature=0.0, max_tokens=10)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "reply": reply.strip(), "latency_ms": round((time.perf_counter() - started) * 1000)}
