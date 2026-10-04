"""Jobs: units of background work run by the worker."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from offscreen.domain.common import JobId, Strict

Lane = Literal["gpu", "cpu", "api"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]

_ALLOWED: dict[JobStatus, frozenset[JobStatus]] = {
    "queued": frozenset({"running", "canceled"}),
    "running": frozenset({"succeeded", "failed", "canceled", "queued"}),  # queued: crash recovery
    "failed": frozenset({"queued"}),  # manual retry
    "succeeded": frozenset(),
    "canceled": frozenset(),
}


class JobCanceled(Exception):
    """Raised by whatever runs a job once it stopped because cancellation was requested."""


def can_transition(src: JobStatus, dst: JobStatus) -> bool:
    return dst in _ALLOWED[src]


class Job(Strict):
    id: JobId
    stage: str
    scope: dict[str, Any] = {}
    lane: Lane
    status: JobStatus = "queued"
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    message: str = ""
    cache_key: str | None = None
    attempt: int = Field(ge=0, default=0)
    error: str | None = None
    log_path: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
