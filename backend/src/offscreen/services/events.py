"""Change detection for the live job feed.

The worker is another process, so the API cannot be told about changes; it looks. A `JobWatcher`
belongs to one open connection: `snapshot()` is what a (re)connecting client gets, `poll()` then
returns the jobs that changed since the last call. Events carry the whole job, so a client simply
replaces what it knows by id, and a reconnect needs no replay."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from offscreen.domain.job import Job
from offscreen.store.models import utcnow
from offscreen.store.repos import JobRepo

RECENT_WINDOW_S = 600.0
"""A fresh connection also learns about jobs that finished while it was away."""

_ACTIVE = ("queued", "running")


def _signature(job: Job) -> tuple[object, ...]:
    return (job.status, job.attempt, job.progress, job.message, job.error)


class JobWatcher:
    def __init__(
        self,
        jobs: JobRepo,
        *,
        window_s: float = RECENT_WINDOW_S,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._jobs = jobs
        self._window = timedelta(seconds=window_s)
        self._clock = clock
        self._seen: dict[str, Job] = {}

    def snapshot(self) -> list[Job]:
        """Active jobs plus those finished within the window; also the baseline for `poll`."""
        jobs = [*self._jobs.active(), *self._jobs.finished_since(self._clock() - self._window)]
        self._seen = {j.id: j for j in jobs}
        return sorted(jobs, key=lambda j: (j.created_at, j.id))

    def poll(self) -> list[Job]:
        """Jobs whose state differs from the last `snapshot` / `poll`, oldest first."""
        changed: list[Job] = []
        active = self._jobs.active()
        active_ids = {j.id for j in active}
        for job in active:
            before = self._seen.get(job.id)
            if before is None or _signature(before) != _signature(job):
                changed.append(job)
            self._seen[job.id] = job
        # Left the active set since last time: finished (or was canceled). Report its end state.
        for job_id, before in list(self._seen.items()):
            if before.status in _ACTIVE and job_id not in active_ids:
                final = self._jobs.get(job_id)
                if final is not None:
                    changed.append(final)
                    self._seen[job_id] = final
                else:
                    del self._seen[job_id]
        return sorted(changed, key=lambda j: (j.created_at, j.id))
