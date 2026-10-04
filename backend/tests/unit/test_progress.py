"""ProgressThrottle and its use by the worker."""

from __future__ import annotations

import time
from pathlib import Path

from offscreen.domain.job import Job
from offscreen.store.db import Database
from offscreen.store.repos import JobRepo
from offscreen.worker import JobCanceled, JobContext, Worker, WorkerSettings
from offscreen.worker.progress import ProgressThrottle


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def make() -> tuple[ProgressThrottle, list[tuple[float, str]], Clock]:
    writes: list[tuple[float, str]] = []
    clock = Clock()
    return ProgressThrottle(lambda f, m: writes.append((f, m)), 0.5, clock), writes, clock


def test_first_report_is_written_at_once() -> None:
    t, writes, _ = make()
    t.report(0.1, "a")
    assert writes == [(0.1, "a")]


def test_burst_of_reports_writes_at_most_twice_per_second() -> None:
    t, writes, clock = make()
    for i in range(1000):  # 1000 reports over one simulated second
        clock.t = 100.0 + i / 1000
        t.report(i / 1000, f"step {i}")
    assert len(writes) <= 3  # t=0, t=0.5, t=1.0 boundary at most
    assert len(writes) >= 2
    assert writes[0] == (0.0, "step 0")


def test_report_inside_the_interval_stays_pending_until_flush_is_due() -> None:
    t, writes, clock = make()
    t.report(0.1, "a")
    clock.t += 0.1
    t.report(0.2, "b")
    assert writes == [(0.1, "a")]
    t.flush()  # not due yet
    assert writes == [(0.1, "a")]
    clock.t += 0.5
    t.flush()
    assert writes == [(0.1, "a"), (0.2, "b")]  # the latest value, not every value


def test_force_flush_writes_the_pending_value_immediately() -> None:
    t, writes, clock = make()
    t.report(0.1, "a")
    clock.t += 0.01
    t.report(0.9, "last")
    t.flush(force=True)
    assert writes[-1] == (0.9, "last")


def test_unchanged_value_is_not_written_again() -> None:
    t, writes, clock = make()
    t.report(0.5, "same")
    clock.t += 1
    t.report(0.5, "same")
    t.flush(force=True)
    assert writes == [(0.5, "same")]


def test_flush_without_pending_is_a_noop() -> None:
    t, writes, _ = make()
    t.flush(force=True)
    assert writes == []


# --- through the worker ----------------------------------------------------------------


def worker_for(tmp_path: Path, execute, interval: float = 0.2):  # type: ignore[no-untyped-def]
    db = Database(tmp_path / "offscreen.db")
    jobs = JobRepo(db)
    settings = WorkerSettings(
        poll_interval_s=0.02, heartbeat_timeout_s=1.0, progress_interval_s=interval
    )
    return db, jobs, Worker(jobs, execute, tmp_path, settings)


def test_hot_loop_does_not_flood_the_database(tmp_path: Path) -> None:
    calls = 0
    db, jobs, _ = worker_for(tmp_path, lambda j, c: None)
    real = jobs.report_progress

    def counting(*a, **k):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real(*a, **k)

    jobs.report_progress = counting  # type: ignore[method-assign]

    def execute(job: Job, ctx: JobContext) -> None:
        end = time.monotonic() + 0.5
        n = 0
        while time.monotonic() < end:
            n += 1
            ctx.progress((n % 100) / 100, f"frame {n}")
            time.sleep(0.0005)

    w = Worker(
        jobs, execute, tmp_path, WorkerSettings(poll_interval_s=0.02, heartbeat_timeout_s=1.0)
    )
    jobs.enqueue("s", {}, "cpu")
    try:
        w.drain()
    finally:
        db.close()
    assert 1 <= calls <= 4  # 0.5 s run at <= 2 writes/s, plus the final flush


def test_last_progress_is_kept_when_a_job_fails_or_is_canceled(tmp_path: Path) -> None:
    def execute(job: Job, ctx: JobContext) -> None:
        ctx.progress(0.1, "early")
        ctx.progress(0.7, "late")  # inside the interval: pending
        raise JobCanceled if job.stage == "c" else RuntimeError("boom")

    db, jobs, w = worker_for(tmp_path, execute, interval=30.0)
    failed = jobs.enqueue("f", {}, "cpu")
    try:
        w.drain()
        got = jobs.get(failed.id)
        assert got is not None
        assert (got.status, got.progress) == ("failed", 0.7)
        jobs.enqueue("c", {}, "cpu")
        w.drain()
        assert jobs.list(status="canceled")[0].progress == 0.7
    finally:
        db.close()


def test_pending_progress_is_flushed_by_the_tick(tmp_path: Path) -> None:
    import threading

    gate = threading.Event()

    def execute(job: Job, ctx: JobContext) -> None:
        ctx.progress(0.1, "early")
        ctx.progress(0.6, "latest")
        gate.wait(5)

    db, jobs, w = worker_for(tmp_path, execute, interval=0.1)
    job = jobs.enqueue("s", {}, "cpu")
    try:
        w.tick()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            w.tick()
            got = jobs.get(job.id)
            if got is not None and got.progress == 0.6:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("pending progress never reached the database")
    finally:
        gate.set()
        w.shutdown(grace_s=2)
        db.close()
