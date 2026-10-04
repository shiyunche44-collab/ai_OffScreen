"""Structured JSON-lines logging with job-context binding and secret scrubbing."""

from __future__ import annotations

import contextvars
import json
import logging
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)
REDACTED = "***"


@contextmanager
def bind_job(job_id: str) -> Iterator[None]:
    """Attach `job_id` to every log record emitted inside the block."""
    token = _job_id.set(job_id)
    try:
        yield
    finally:
        _job_id.reset(token)


class _Scrub(logging.Filter):
    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = sorted({s for s in secrets if len(s) >= 8}, key=len, reverse=True)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for s in self._secrets:
            msg = msg.replace(s, REDACTED)
        record.msg, record.args = msg, None
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if (job := _job_id.get()) is not None:
            entry["job_id"] = job
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            entry.update(extra)
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(
    log_dir: Path | None = None,
    level: int = logging.INFO,
    secret_env_names: Iterable[str] = (),
) -> None:
    """Configure the `offscreen` logger. Values of the named env vars are scrubbed."""
    logger = logging.getLogger("offscreen")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False
    scrub = _Scrub(os.environ.get(n, "") for n in secret_env_names)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "offscreen.log", encoding="utf-8"))
    for h in handlers:
        h.setFormatter(_JsonFormatter())
        h.addFilter(scrub)
        logger.addHandler(h)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"offscreen.{name}")
