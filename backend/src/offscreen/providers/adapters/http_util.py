"""Shared HTTP failure handling for provider adapters: which errors are worth retrying,
how long to wait, and MiniMax's `base_resp` error codes (docs/PROVIDERS.md)."""

from __future__ import annotations

import random
import re
from typing import Any, Literal

MAX_RETRIES = 3  # transient failures only; so up to 4 requests per chat
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0

FailureKind = Literal["quota", "auth", "transient", "fatal"]

# MiniMax reports errors in `base_resp`, sometimes with HTTP 200.
_MINIMAX_QUOTA = {1008, 2056}  # insufficient balance; token-plan usage window exhausted
_MINIMAX_AUTH = {1004, 2049}
_MINIMAX_TRANSIENT = {1000, 1001, 1002, 1013, 1033, 1039, 1041}  # timeout, RPM/TPM, internal
# Deliberately excludes plain "rate limit", which is transient.
_QUOTA_WORDS = re.compile(
    r"insufficient|balance|quota|usage[ _]limit|exhaust|billing|额度|余额|用量", re.IGNORECASE
)


def classify_failure(status: int, body: Any, text: str = "") -> FailureKind:
    """Decide how to react to an error response. `body` is the parsed JSON, if any."""
    base = body.get("base_resp") if isinstance(body, dict) else None
    code = base.get("status_code") if isinstance(base, dict) else None
    if isinstance(code, int) and code != 0:
        if code in _MINIMAX_QUOTA:
            return "quota"
        if code in _MINIMAX_AUTH:
            return "auth"
        if code in _MINIMAX_TRANSIENT:
            return "transient"
    if status in (401, 403):
        return "auth"
    if status == 402:  # DeepSeek: Insufficient Balance
        return "quota"
    if status == 429:
        return "quota" if _QUOTA_WORDS.search(text) else "transient"
    if status in (408, 409, 425) or status >= 500:
        return "transient"
    if isinstance(code, int) and code != 0:
        return "fatal"
    return "fatal"


def error_text(body: Any, text: str) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        base = body.get("base_resp")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:300]
        if isinstance(base, dict) and base.get("status_msg"):
            return f"{base.get('status_code')} {base['status_msg']}"[:300]
    return text[:300]


def parse_retry_after(value: str | None) -> float | None:
    try:
        return min(float(value), BACKOFF_CAP_S) if value else None
    except ValueError:
        return None


def backoff_delay(attempt: int) -> float:
    delay: float = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2**attempt)
    return delay * (0.75 + random.random() / 2)
