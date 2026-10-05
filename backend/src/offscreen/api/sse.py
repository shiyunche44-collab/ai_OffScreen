"""Server-Sent Events plumbing: the wire format and the job feed loop."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from offscreen.services.events import JobWatcher


@dataclass(frozen=True)
class EventSettings:
    poll_s: float = 0.5
    """How often the database is looked at for changes (progress is written at most twice a
    second, so faster polling would find nothing new)."""
    keepalive_s: float = 15.0
    """Idle connections get a comment this often, so proxies do not drop them."""
    max_lifetime_s: float = 300.0
    """A connection is closed after this long. The browser reconnects by itself and receives a
    fresh snapshot, so whatever a long-lived connection might have missed is repaired."""
    retry_ms: int = 2000
    """Reconnect delay suggested to the browser."""


def format_event(event: str, data: str) -> str:
    """One SSE message. `data` must be a single line (JSON from pydantic is)."""
    if "\n" in data or "\r" in data:
        raise ValueError("SSE data must be a single line")
    return f"event: {event}\ndata: {data}\n\n"


async def job_events(
    watcher: JobWatcher,
    is_disconnected: Callable[[], Awaitable[bool]],
    settings: EventSettings,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """`snapshot` first (`{"jobs": [...]}`), then one `job` event (a full Job) per change."""
    yield f"retry: {settings.retry_ms}\n\n"
    jobs = await asyncio.to_thread(watcher.snapshot)
    yield format_event("snapshot", '{"jobs":[' + ",".join(j.model_dump_json() for j in jobs) + "]}")

    started = last_sent = clock()
    while clock() - started < settings.max_lifetime_s:
        await asyncio.sleep(settings.poll_s)
        if await is_disconnected():
            return
        changed = await asyncio.to_thread(watcher.poll)
        for job in changed:
            yield format_event("job", job.model_dump_json())
        if changed:
            last_sent = clock()
        elif clock() - last_sent >= settings.keepalive_s:
            yield ": keep-alive\n\n"
            last_sent = clock()
