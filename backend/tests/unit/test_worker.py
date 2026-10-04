"""Worker: lanes, retries, cancellation, crash recovery, with a fake executor."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from offscreen.domain.job import Job
from offscreen.providers.ports import LLMAuthError, LLMRateLimited, TTSRateLimited
from offscreen.store.db import Database
from offscreen.store.models import utcnow
from offscreen.store.repos import JobRepo
from offscreen.worker import JobCanceled, JobContext, Worker, WorkerSettings

FAST = WorkerSettings(
    poll_interval_s=0.02, heartbeat_timeout_s=1.0, retry_base_s=0.02, retry_max_s=0.05
)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    d = Database(tmp_path / "offscreen.db")
    yield d
    d.close()


@pytest.fixture
def jobs(db: Database) -> JobRepo:
    return JobRepo(db)


def make(
    jobs: JobRepo,
    tmp_path: Path,
    execute: Callable[[Job, JobContext], None],
    settings: WorkerSettings = FAST,
    **kw: object,
) -> Worker:
    return Worker(jobs, execute, tmp_path, settings, **kw)  # type: ignore[arg-type]


def wait_for(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "condition not reached in time"
        time.sleep(0.01)


def status(jobs: JobRepo, job_id: str) -> str:
    j = jobs.get(job_id)
    assert j is not None
    return j.status


# --- success and failure ----------------------------------------------------------------


def test_runs_a_job_to_success_and_logs_it(jobs: JobRepo, tmp_path: Path) -> None:
    seen: list[tuple[str, dict[str, str]]] = []

    def execute(job: Job, ctx: JobContext) -> None:
        seen.append((job.stage, job.scope))  # type: ignore[arg-type]
        ctx.progress(0.5, "halfway")
        ctx.log("custom line")

    job = jobs.enqueue("analysis.shots", {"asset_id": "ast_1"}, "cpu")
    make(jobs, tmp_path, execute).drain()

    done = jobs.get(job.id)
    assert done is not None
    assert (done.status, done.progress, done.message, done.attempt) == ("succeeded", 1.0, "done", 1)
    assert done.started_at and done.finished_at
    assert seen == [("analysis.shots", {"asset_id": "ast_1"})]
    log = (tmp_path / "logs" / f"{job.id}.log").read_text(encoding="utf-8")
    assert "start analysis.shots" in log and "custom line" in log and "succeeded" in log


def test_failure_is_recorded_with_error_and_traceback_and_not_retried(
    jobs: JobRepo, tmp_path: Path
) -> None:
    calls = []

    def execute(job: Job, ctx: JobContext) -> None:
        calls.append(job.attempt)
        raise ValueError("bad input")

    job = jobs.enqueue("s", {}, "cpu")
    make(jobs, tmp_path, execute).drain()

    failed = jobs.get(job.id)
    assert failed is not None
    assert failed.status == "failed" and failed.error == "ValueError: bad input"
    assert calls == [1]  # cpu lane: no automatic retry, even for retryable-looking errors
    assert "Traceback" in (tmp_path / "logs" / f"{job.id}.log").read_text(encoding="utf-8")

    jobs.retry(job.id)  # a person decides
    make(jobs, tmp_path, lambda j, c: None).drain()
    assert status(jobs, job.id) == "succeeded"


def test_cpu_lane_does_not_auto_retry_rate_limits(jobs: JobRepo, tmp_path: Path) -> None:
    def execute(job: Job, ctx: JobContext) -> None:
        raise LLMRateLimited("429")

    job = jobs.enqueue("s", {}, "cpu")
    make(jobs, tmp_path, execute).drain()
    assert status(jobs, job.id) == "failed"


# --- lanes ------------------------------------------------------------------------------


class Gauge:
    """Tracks how many executors run at once."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0

    def __call__(self, job: Job, ctx: JobContext) -> None:
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.1)
        with self.lock:
            self.now -= 1


def test_gpu_lane_runs_one_job_at_a_time(jobs: JobRepo, tmp_path: Path) -> None:
    gauge = Gauge()
    ids = [jobs.enqueue("s", {}, "gpu").id for _ in range(3)]
    make(jobs, tmp_path, gauge).drain()
    assert gauge.peak == 1
    assert all(status(jobs, i) == "succeeded" for i in ids)


def test_cpu_lane_is_limited_by_its_setting(jobs: JobRepo, tmp_path: Path) -> None:
    gauge = Gauge()
    for _ in range(6):
        jobs.enqueue("s", {}, "cpu")
    settings = WorkerSettings(
        lane_limits={"gpu": 1, "cpu": 3, "api": 4}, poll_interval_s=0.02, heartbeat_timeout_s=1.0
    )
    make(jobs, tmp_path, gauge, settings).drain()
    assert gauge.peak == 3


def test_lanes_run_in_parallel_with_each_other(jobs: JobRepo, tmp_path: Path) -> None:
    gauge = Gauge()
    for lane in ("gpu", "cpu", "api"):
        jobs.enqueue("s", {}, lane)  # type: ignore[arg-type]
    make(jobs, tmp_path, gauge).drain()
    assert gauge.peak == 3


def test_gpu_limit_other_than_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="gpu"):
        WorkerSettings(lane_limits={"gpu": 2})
    with pytest.raises(ValueError, match="at least 1"):
        WorkerSettings(lane_limits={"cpu": 0})
    with pytest.raises(ValueError, match="heartbeat"):
        WorkerSettings(poll_interval_s=5, heartbeat_timeout_s=5)


# --- api retries ------------------------------------------------------------------------


def test_api_lane_retries_rate_limits_with_backoff_then_succeeds(
    jobs: JobRepo, tmp_path: Path
) -> None:
    attempts: list[int] = []

    def execute(job: Job, ctx: JobContext) -> None:
        attempts.append(job.attempt)
        if job.attempt < 3:
            raise TTSRateLimited("still limited")

    job = jobs.enqueue("s", {}, "api")
    make(jobs, tmp_path, execute).drain()

    done = jobs.get(job.id)
    assert done is not None and done.status == "succeeded"
    assert attempts == [1, 2, 3]
    assert done.error is None  # cleared by the successful run's finish


def test_api_lane_gives_up_after_three_retries(jobs: JobRepo, tmp_path: Path) -> None:
    attempts: list[int] = []

    def execute(job: Job, ctx: JobContext) -> None:
        attempts.append(job.attempt)
        raise LLMRateLimited("429")

    job = jobs.enqueue("s", {}, "api")
    make(jobs, tmp_path, execute).drain()

    failed = jobs.get(job.id)
    assert failed is not None and failed.status == "failed"
    assert failed.error == "LLMRateLimited: 429"
    assert attempts == [1, 2, 3, 4]  # the first run plus three retries


def test_api_lane_does_not_retry_auth_or_other_errors(jobs: JobRepo, tmp_path: Path) -> None:
    attempts: list[int] = []

    def execute(job: Job, ctx: JobContext) -> None:
        attempts.append(job.attempt)
        raise LLMAuthError("bad key")

    job = jobs.enqueue("s", {}, "api")
    make(jobs, tmp_path, execute).drain()
    assert attempts == [1] and status(jobs, job.id) == "failed"


def test_backoff_grows_exponentially_and_is_capped() -> None:
    s = WorkerSettings(retry_base_s=2.0, retry_max_s=10.0)
    assert [s.backoff_s(n) for n in (1, 2, 3, 4, 5)] == [2.0, 4.0, 8.0, 10.0, 10.0]


def test_retry_waits_for_the_backoff(jobs: JobRepo, tmp_path: Path) -> None:
    stamps: list[float] = []

    def execute(job: Job, ctx: JobContext) -> None:
        stamps.append(time.monotonic())
        if job.attempt == 1:
            raise LLMRateLimited("429")

    settings = WorkerSettings(poll_interval_s=0.02, heartbeat_timeout_s=1.0, retry_base_s=0.3)
    jobs.enqueue("s", {}, "api")
    make(jobs, tmp_path, execute, settings).drain()
    assert stamps[1] - stamps[0] >= 0.25


# --- cancellation -----------------------------------------------------------------------


def test_cancel_while_running_stops_the_executor_cooperatively(
    jobs: JobRepo, tmp_path: Path
) -> None:
    started = threading.Event()

    def execute(job: Job, ctx: JobContext) -> None:
        started.set()
        while not ctx.is_canceled():
            time.sleep(0.01)
        raise JobCanceled

    job = jobs.enqueue("s", {}, "cpu")
    w = make(jobs, tmp_path, execute)
    w.tick()
    assert started.wait(2)
    assert status(jobs, job.id) == "running"

    jobs.request_cancel(job.id)
    w.drain()
    canceled = jobs.get(job.id)
    assert canceled is not None and canceled.status == "canceled" and canceled.error is None


def test_error_raised_because_of_a_cancel_counts_as_canceled(jobs: JobRepo, tmp_path: Path) -> None:
    """E.g. ffmpeg was killed and the stage surfaced a generic error."""
    started = threading.Event()

    def execute(job: Job, ctx: JobContext) -> None:
        started.set()
        while not ctx.is_canceled():
            time.sleep(0.01)
        raise RuntimeError("subprocess died")

    job = jobs.enqueue("s", {}, "cpu")
    w = make(jobs, tmp_path, execute)
    w.tick()
    assert started.wait(2)
    jobs.request_cancel(job.id)
    w.drain()
    assert status(jobs, job.id) == "canceled"


def test_cancel_queued_job_never_runs(jobs: JobRepo, tmp_path: Path) -> None:
    ran = []
    job = jobs.enqueue("s", {}, "cpu")
    jobs.request_cancel(job.id)
    make(jobs, tmp_path, lambda j, c: ran.append(j.id)).drain()
    assert ran == [] and status(jobs, job.id) == "canceled"


def test_finishing_despite_a_late_cancel_request_is_success(jobs: JobRepo, tmp_path: Path) -> None:
    """Cancel is a request: work that completes anyway keeps its result."""
    job = jobs.enqueue("s", {}, "cpu")

    def execute(j: Job, ctx: JobContext) -> None:
        jobs.request_cancel(j.id)

    make(jobs, tmp_path, execute).drain()
    assert status(jobs, job.id) == "succeeded"


# --- crash recovery and shutdown --------------------------------------------------------


def test_job_orphaned_by_a_crashed_worker_is_requeued_and_completed(
    jobs: JobRepo, tmp_path: Path
) -> None:
    job = jobs.enqueue("s", {}, "cpu")
    t0 = utcnow()
    ghost = jobs.claim("cpu", 2, now=t0)  # a worker claimed it, then was killed
    assert ghost is not None and status(jobs, job.id) == "running"

    ran = []
    w = make(jobs, tmp_path, lambda j, c: ran.append(j.attempt))
    w.tick()  # heartbeat is fresh: nothing to recover yet
    assert w.active == [] and status(jobs, job.id) == "running"

    later = t0 + timedelta(seconds=5)
    w2 = make(jobs, tmp_path, lambda j, c: ran.append(j.attempt), clock=lambda: later)
    w2.drain()
    assert status(jobs, job.id) == "succeeded"
    assert ran == [2]


def test_a_stalled_run_that_lost_its_job_does_not_clobber_the_replacement(
    jobs: JobRepo, tmp_path: Path
) -> None:
    release = threading.Event()
    results: list[str] = []

    def execute(job: Job, ctx: JobContext) -> None:
        if job.attempt == 1:
            release.wait(5)  # stalls (as if the process was frozen)
            results.append("old finished")
        else:
            results.append("new finished")

    job = jobs.enqueue("s", {}, "cpu")
    w = make(jobs, tmp_path, execute)
    w.tick()
    wait_for(lambda: w.active == [job.id])

    # Another worker decides the heartbeat is stale and takes over.
    jobs.requeue_stale(1.0, now=utcnow() + timedelta(seconds=10))
    new = jobs.claim("cpu", 1)
    assert new is not None and new.attempt == 2

    release.set()
    wait_for(lambda: results == ["old finished"])
    wait_for(lambda: w.active == [])
    assert status(jobs, job.id) == "running"  # the old run's result was dropped
    assert jobs.finish(new.id, new.attempt, "succeeded")


def test_shutdown_returns_unfinished_jobs_to_the_queue(jobs: JobRepo, tmp_path: Path) -> None:
    started = threading.Event()

    def execute(job: Job, ctx: JobContext) -> None:
        started.set()
        while not ctx.is_canceled():
            time.sleep(0.01)
        raise JobCanceled

    job = jobs.enqueue("s", {}, "cpu")
    w = make(jobs, tmp_path, execute)
    w.tick()
    assert started.wait(2)
    w.shutdown(grace_s=2)

    queued = jobs.get(job.id)
    assert queued is not None and queued.status == "queued"
    assert queued.message == "requeued: worker stopped"

    # The next worker picks it up again.
    make(jobs, tmp_path, lambda j, c: None).drain()
    assert status(jobs, job.id) == "succeeded"


def test_run_forever_serves_jobs_until_stopped(jobs: JobRepo, tmp_path: Path) -> None:
    w = make(jobs, tmp_path, lambda j, c: None)
    t = threading.Thread(target=w.run_forever)
    t.start()
    try:
        ids = [jobs.enqueue("s", {}, "cpu").id for _ in range(3)]  # enqueued while it runs
        wait_for(lambda: all(status(jobs, i) == "succeeded" for i in ids))
    finally:
        w.stop()
        t.join(5)
    assert not t.is_alive()


def test_drain_times_out_when_a_job_cannot_finish(jobs: JobRepo, tmp_path: Path) -> None:
    release = threading.Event()
    jobs.enqueue("s", {}, "cpu")
    w = make(jobs, tmp_path, lambda j, c: release.wait(5))
    try:
        with pytest.raises(TimeoutError):
            w.drain(timeout_s=0.2)
    finally:
        release.set()
        w.shutdown(grace_s=2)
