"""The worker: claims queued jobs lane by lane and runs them on threads.

It knows nothing about stages. What a job *does* is the injected `execute` callable (wired to
services by the caller), which makes the scheduling, retry and recovery rules testable with a
fake. The rules (ARCHITECTURE §3.3, §6.4):

- per-lane concurrency, gpu fixed at 1; the limit is enforced by the database claim, so it also
  holds with several worker processes
- every tick each running job gets a heartbeat; jobs whose heartbeat went stale (a crashed
  worker) are put back in the queue
- api lane: rate-limit style failures are retried with exponential backoff, at most
  `max_api_retries` times; gpu / cpu failures are final, a person decides about a retry
- cancellation is cooperative: the flag is picked up on the next tick and shows as
  `ctx.is_canceled()`; the executor stops at its next checkpoint
"""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from offscreen.domain.job import Job, JobCanceled, Lane
from offscreen.log import bind_job
from offscreen.providers.ports import LLMRateLimited, TTSRateLimited
from offscreen.store.models import utcnow
from offscreen.store.repos import JobRepo
from offscreen.worker.progress import DEFAULT_INTERVAL_S, ProgressThrottle

logger = logging.getLogger(__name__)

LANES: tuple[Lane, ...] = ("gpu", "cpu", "api")
MAX_ERROR_CHARS = 2000


def is_retryable(exc: BaseException) -> bool:
    """Failures worth another attempt: still rate limited (or 5xx / network) after the
    adapter's own backoff. Auth errors, exhausted quota and bad replies are not."""
    return isinstance(exc, LLMRateLimited | TTSRateLimited)


@dataclass(frozen=True)
class WorkerSettings:
    lane_limits: Mapping[Lane, int] = field(default_factory=lambda: {"gpu": 1, "cpu": 2, "api": 4})
    poll_interval_s: float = 1.0
    """Tick period: claim new jobs, heartbeat running ones, notice cancel requests."""
    heartbeat_timeout_s: float = 30.0
    """A running job whose last heartbeat is older than this is considered orphaned."""
    max_api_retries: int = 3
    retry_base_s: float = 2.0
    retry_max_s: float = 60.0
    progress_interval_s: float = DEFAULT_INTERVAL_S
    """Job progress reaches the database at most once per interval."""

    def __post_init__(self) -> None:
        if self.lane_limits.get("gpu", 1) != 1:
            raise ValueError("the gpu lane runs one job at a time (ARCHITECTURE §3.3)")
        if any(n < 1 for n in self.lane_limits.values()):
            raise ValueError("lane limits must be at least 1")
        if self.heartbeat_timeout_s <= self.poll_interval_s * 2:
            raise ValueError("heartbeat_timeout_s must leave room for several ticks")

    def limit(self, lane: Lane) -> int:
        return self.lane_limits.get(lane, 1)

    def backoff_s(self, attempt: int) -> float:
        """Delay before the retry that follows run number `attempt` (1-based)."""
        return float(min(self.retry_max_s, self.retry_base_s * 2 ** (attempt - 1)))


class JobContext:
    """What an executor sees of its own job."""

    def __init__(
        self,
        job: Job,
        run: _Run,
        repo: JobRepo,
        log_file: Path | None,
        progress_interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        self.job = job
        self._run = run
        self._log_file = log_file
        self._progress = ProgressThrottle(
            lambda frac, msg: repo.report_progress(job.id, job.attempt, frac, msg),
            progress_interval_s,
        )

    def progress(self, frac: float, msg: str = "") -> None:
        self._progress.report(frac, msg)

    def flush_progress(self, *, force: bool = False) -> None:
        self._progress.flush(force=force)

    def is_canceled(self) -> bool:
        return self._run.stop.is_set()

    def log(self, msg: str) -> None:
        _append_log(self._log_file, msg)


Executor = Callable[[Job, JobContext], None]


class _Run:
    """Bookkeeping for one running job."""

    def __init__(self, job: Job) -> None:
        self.job = job
        self.stop = threading.Event()
        self.reason: Literal["cancel", "shutdown", "lost"] = "cancel"
        self.thread: threading.Thread | None = None
        self.ctx: JobContext | None = None


class Worker:
    def __init__(
        self,
        jobs: JobRepo,
        execute: Executor,
        data_dir: Path,
        settings: WorkerSettings | None = None,
        *,
        retryable: Callable[[BaseException], bool] = is_retryable,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.jobs = jobs
        self.execute = execute
        self.data_dir = data_dir
        self.settings = settings or WorkerSettings()
        self._retryable = retryable
        self._clock = clock
        self._lock = threading.Lock()
        self._runs: dict[str, _Run] = {}
        self._stopped = threading.Event()

    # ---- loop --------------------------------------------------------------------------
    def tick(self) -> int:
        """One scheduling round; returns how many jobs were started."""
        now = self._clock()
        self._heartbeat(now)
        requeued = self.jobs.requeue_stale(self.settings.heartbeat_timeout_s, now=now)
        if requeued:
            logger.warning("requeued %d job(s) with a stale heartbeat: %s", len(requeued), requeued)
        started = 0
        for lane in LANES:
            while (job := self.jobs.claim(lane, self.settings.limit(lane), now=now)) is not None:
                self._start(job)
                started += 1
        return started

    def run_forever(self) -> None:
        """Tick until `stop()`; then give running jobs back to the queue (see `shutdown`)."""
        logger.info("worker started, lanes %s", dict(self.settings.lane_limits))
        while not self._stopped.is_set():
            try:
                self.tick()
            except Exception:  # a locked database or similar must not kill the loop
                logger.exception("worker tick failed")
            self._stopped.wait(self.settings.poll_interval_s)
        self.shutdown()

    def stop(self) -> None:
        """Ask `run_forever` to return (safe from signal handlers and other threads)."""
        self._stopped.set()

    def shutdown(self, grace_s: float = 10.0) -> None:
        """Stop running jobs cooperatively and put the ones that did not finish back in the
        queue, so the next worker resumes them."""
        self._stopped.set()
        with self._lock:
            runs = list(self._runs.values())
        for run in runs:
            run.reason = "shutdown"
            run.stop.set()
        deadline = self._clock() + timedelta(seconds=grace_s)
        for run in runs:
            if run.thread is not None:
                run.thread.join(max(0.0, (deadline - self._clock()).total_seconds()))

    def drain(self, timeout_s: float = 60.0) -> None:
        """Tick until nothing is queued or running (tests, one-shot CLI runs)."""
        end = self._clock() + timedelta(seconds=timeout_s)
        while True:
            self.tick()
            with self._lock:
                busy = bool(self._runs)
            if not busy and self.jobs.count("queued") == 0 and self.jobs.count("running") == 0:
                return
            if self._clock() > end:
                raise TimeoutError("worker did not drain in time")
            self._stopped.wait(min(self.settings.poll_interval_s, 0.05))

    @property
    def active(self) -> list[str]:
        with self._lock:
            return list(self._runs)

    # ---- internals ---------------------------------------------------------------------
    def _heartbeat(self, now: datetime) -> None:
        with self._lock:
            runs = list(self._runs.values())
        for run in runs:
            if run.ctx is not None:
                run.ctx.flush_progress()
            flag = self.jobs.heartbeat(run.job.id, run.job.attempt, now=now)
            if flag is None:  # requeued or finished elsewhere: this run no longer counts
                run.reason = "lost"
                run.stop.set()
            elif flag:
                run.stop.set()

    def _start(self, job: Job) -> None:
        run = _Run(job)
        thread = threading.Thread(target=self._run, args=(run,), name=f"job-{job.id}", daemon=True)
        run.thread = thread
        with self._lock:
            self._runs[job.id] = run
        thread.start()

    def _log_file(self, job: Job) -> Path | None:
        return self.data_dir / job.log_path if job.log_path else None

    def _run(self, run: _Run) -> None:
        job = run.job
        log = self._log_file(job)
        ctx = JobContext(job, run, self.jobs, log, self.settings.progress_interval_s)
        run.ctx = ctx
        try:
            with bind_job(job.id):
                _append_log(log, f"start {job.stage} {job.scope} (attempt {job.attempt})")
                try:
                    try:
                        self.execute(job, ctx)
                    finally:  # the last progress survives a failure or cancel
                        ctx.flush_progress(force=True)
                except JobCanceled:
                    self._interrupted(run, "canceled", None)
                except Exception as exc:
                    self._failed(run, exc, log)
                else:
                    if self.jobs.finish(job.id, job.attempt, "succeeded", message="done"):
                        _append_log(log, "succeeded")
        except Exception:  # bookkeeping failed; the heartbeat timeout will requeue the job
            logger.exception("job %s: could not record its outcome", job.id)
        finally:
            with self._lock:
                self._runs.pop(job.id, None)

    def _interrupted(self, run: _Run, reason: str, error: str | None) -> None:
        job = run.job
        if run.reason == "shutdown":
            self.jobs.reschedule(
                job.id, job.attempt, not_before=None, error=None, message="requeued: worker stopped"
            )
        elif run.reason == "cancel":
            self.jobs.finish(job.id, job.attempt, "canceled", error=error, message="canceled")
        # "lost": another run owns the job now; the guarded updates above would do nothing
        _append_log(self._log_file(job), f"{reason} ({run.reason})")

    def _failed(self, run: _Run, exc: Exception, log: Path | None) -> None:
        job = run.job
        _append_log(log, "".join(traceback.format_exception(exc)).rstrip())
        error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        if run.stop.is_set():  # it died because we told it to stop (e.g. ffmpeg was killed)
            self._interrupted(run, "stopped with error", error)
            return
        retries_used = job.attempt - 1
        if (
            job.lane == "api"
            and self._retryable(exc)
            and retries_used < self.settings.max_api_retries
        ):
            delay = self.settings.backoff_s(job.attempt)
            when = self._clock() + timedelta(seconds=delay)
            msg = f"retry {job.attempt}/{self.settings.max_api_retries} in {delay:g}s"
            if self.jobs.reschedule(job.id, job.attempt, not_before=when, error=error, message=msg):
                _append_log(log, msg)
            return
        self.jobs.finish(job.id, job.attempt, "failed", error=error, message="failed")
        logger.warning("job %s failed: %s", job.id, error)


def _append_log(path: Path | None, msg: str) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{utcnow().isoformat(timespec='milliseconds')}Z {msg}\n")
    except OSError:  # a full disk must not fail the job
        logger.warning("cannot write job log %s", path)
