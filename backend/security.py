"""Rate limiting and admin authentication."""

import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Header, HTTPException, Request

from config import settings

ADMIN_TOKEN_TTL_SECONDS = 12 * 60 * 60
LOGIN_ATTEMPTS_PER_MINUTE = 5


class RateLimiter:
    """In-memory sliding-window limiter keyed by client (single-process deployments)."""

    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True


_limiter = RateLimiter()
_admin_tokens: Dict[str, float] = {}
_tokens_lock = threading.Lock()


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def chat_rate_limit(request: Request) -> None:
    """FastAPI dependency limiting chat requests per client per minute."""
    if not _limiter.allow(f"chat:{_client_key(request)}", max(1, settings.RATE_LIMIT_PER_MINUTE)):
        raise HTTPException(status_code=429, detail="تعداد درخواست‌ها زیاد است. لطفاً یک دقیقه دیگر دوباره تلاش کنید.")


def login(request: Request, password: str) -> str:
    """Return a new admin session token for a correct password."""
    if not settings.ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="ADMIN_PASSWORD is not set in backend/.env; the admin panel is disabled.")
    if not _limiter.allow(f"login:{_client_key(request)}", LOGIN_ATTEMPTS_PER_MINUTE):
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again in a minute.")
    if not hmac.compare_digest(password.encode("utf-8"), settings.ADMIN_PASSWORD.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Wrong password.")
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _tokens_lock:
        for expired in [t for t, expiry in _admin_tokens.items() if expiry < now]:
            del _admin_tokens[expired]
        _admin_tokens[token] = now + ADMIN_TOKEN_TTL_SECONDS
    return token


def logout(token: str) -> None:
    with _tokens_lock:
        _admin_tokens.pop(token, None)


def require_admin(authorization: str = Header(default="")) -> str:
    """FastAPI dependency: a valid 'Authorization: Bearer <token>' header from /admin/login."""
    token = authorization.removeprefix("Bearer ").strip()
    with _tokens_lock:
        expiry = _admin_tokens.get(token)
    if not token or expiry is None or expiry < time.time():
        raise HTTPException(status_code=401, detail="Admin login required.")
    return token
