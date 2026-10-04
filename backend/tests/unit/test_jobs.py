"""JobRepo: the queue's state machine and its atomicity guarantees."""

from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path

import pytest

from offscreen.store.db import Database
from offscreen.store.models import utcnow
from offscreen.store.repos import JobRepo


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "data" / "offscreen.db")
    yield d  # type: ignore[misc]
    d.close()


@pytest.fixture
def jobs(db: Database) -> JobRepo:
    return JobRepo(db)


def test_enqueue_creates_a_queued_job_with_a_log_path(jobs: JobRepo) -> None:
    job = jobs.enqueue("analysis.shots", {"asset_id": "ast_1"}, "cpu")
    assert job.id.startswith("job_")
    assert (job.status, job.attempt, job.progress) == ("queued", 0, 0.0)
    assert job.scope == {"asset_id": "ast_1"}
    assert job.log_path == f"logs/{job.id}.log"
    assert jobs.get(job.id) == job
    assert jobs.get("job_missing") is None


def test_list_is_newest_first_and_filters(jobs: JobRepo) -> None:
    a = jobs.enqueue("s", {}, "cpu")
    b = jobs.enqueue("s", {}, "api")
    c = jobs.enqueue("s", {}, "cpu")
    assert [j.id for j in jobs.list()] == [c.id, b.id, a.id]
    assert [j.id for j in jobs.list(lane="cpu")] == [c.id, a.id]
    assert [j.id for j in jobs.list(status="running")] == []
    assert len(jobs.list(limit=2)) == 2


def test_claim_takes_oldest_of_the_lane_and_marks_it_running(jobs: JobRepo) -> None:
    first = jobs.enqueue("s", {}, "cpu")
    jobs.enqueue("s", {}, "api")
    second = jobs.enqueue("s", {}, "cpu")

    got = jobs.claim("cpu", 5)
    assert got is not None and got.id == first.id
    assert (got.status, got.attempt) == ("running", 1)
    assert got.started_at is not None
    assert jobs.claim("cpu", 5).id == second.id  # type: ignore[union-attr]
    assert jobs.claim("cpu", 5) is None  # the api job belongs to another lane
    assert jobs.claim("gpu", 1) is None


def test_claim_respects_the_lane_concurrency_limit(jobs: JobRepo) -> None:
    for _ in range(3):
        jobs.enqueue("s", {}, "gpu")
    first = jobs.claim("gpu", 1)
    assert first is not None
    assert jobs.claim("gpu", 1) is None  # one gpu job is running
    assert jobs.finish(first.id, first.attempt, "succeeded")
    assert jobs.claim("gpu", 1) is not None  # slot freed
    assert jobs.claim("gpu", 2) is not None  # a higher limit admits another


def test_claim_skips_jobs_waiting_for_their_backoff(jobs: JobRepo) -> None:
    job = jobs.enqueue("s", {}, "api")
    run = jobs.claim("api", 1)
    assert run is not None
    later = utcnow() + timedelta(seconds=30)
    assert jobs.reschedule(run.id, run.attempt, not_before=later, error="429", message="retrying")

    queued = jobs.get(job.id)
    assert queued is not None and (queued.status, queued.error) == ("queued", "429")
    assert jobs.claim("api", 1) is None
    again = jobs.claim("api", 1, now=later + timedelta(seconds=1))
    assert again is not None and again.attempt == 2


def test_concurrent_claims_never_hand_out_a_job_twice(jobs: JobRepo) -> None:
    ids = {jobs.enqueue("s", {}, "cpu").id for _ in range(40)}
    got: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        while (j := jobs.claim("cpu", 1000)) is not None:
            with lock:
                got.append(j.id)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(got) == sorted(ids)  # every job exactly once


def test_claims_from_two_connections_respect_the_limit(tmp_path: Path) -> None:
    """Two worker processes = two Database objects on one file."""
    path = tmp_path / "x.db"
    a, b = Database(path), Database(path)
    try:
        ja, jb = JobRepo(a), JobRepo(b)
        for _ in range(4):
            ja.enqueue("s", {}, "gpu")
        assert ja.claim("gpu", 1) is not None
        assert jb.claim("gpu", 1) is None
    finally:
        a.close()
        b.close()


def test_heartbeat_reports_cancel_requests_and_lost_ownership(jobs: JobRepo) -> None:
    jobs.enqueue("s", {}, "cpu")
    run = jobs.claim("cpu", 1)
    assert run is not None
    assert jobs.heartbeat(run.id, run.attempt) is False
    jobs.request_cancel(run.id)
    assert jobs.heartbeat(run.id, run.attempt) is True
    assert jobs.heartbeat(run.id, run.attempt + 1) is None  # not our run
    assert jobs.heartbeat("job_missing", 1) is None


def test_finish_states_and_ownership_guard(jobs: JobRepo) -> None:
    jobs.enqueue("s", {}, "cpu")
    run = jobs.claim("cpu", 1)
    assert run is not None
    assert not jobs.finish(run.id, run.attempt + 1, "succeeded")  # someone else's run
    assert jobs.finish(run.id, run.attempt, "failed", error="boom")
    done = jobs.get(run.id)
    assert done is not None
    assert (done.status, done.error, done.finished_at is not None) == ("failed", "boom", True)
    assert not jobs.finish(run.id, run.attempt, "succeeded")  # already finished
    with pytest.raises(ValueError):
        jobs.finish(run.id, run.attempt, "queued")


def test_success_sets_progress_to_one(jobs: JobRepo) -> None:
    jobs.enqueue("s", {}, "cpu")
    run = jobs.claim("cpu", 1)
    assert run is not None
    jobs.report_progress(run.id, run.attempt, 0.4, "halfway")
    mid = jobs.get(run.id)
    assert mid is not None and (mid.progress, mid.message) == (0.4, "halfway")
    jobs.report_progress(run.id, run.attempt, 7.0, "clamped")
    assert jobs.get(run.id).progress == 1.0  # type: ignore[union-attr]
    jobs.report_progress(run.id, run.attempt + 1, 0.1, "stale run")  # ignored
    assert jobs.get(run.id).message == "clamped"  # type: ignore[union-attr]
    assert jobs.finish(run.id, run.attempt, "succeeded", message="done")
    assert jobs.get(run.id).message == "done"  # type: ignore[union-attr]


def test_cancel_queued_is_immediate_and_running_is_a_request(jobs: JobRepo) -> None:
    queued = jobs.enqueue("s", {}, "cpu")
    canceled = jobs.request_cancel(queued.id)
    assert canceled is not None and canceled.status == "canceled"
    assert jobs.claim("cpu", 1) is None

    jobs.enqueue("s", {}, "cpu")
    run = jobs.claim("cpu", 1)
    assert run is not None
    after = jobs.request_cancel(run.id)
    assert after is not None and after.status == "running"  # stage must stop cooperatively
    assert jobs.finish(run.id, run.attempt, "canceled")

    assert jobs.request_cancel(run.id).status == "canceled"  # type: ignore[union-attr]  # no-op
    assert jobs.request_cancel("job_missing") is None


def test_retry_only_from_failed_and_resets_the_budget(jobs: JobRepo) -> None:
    jobs.enqueue("s", {}, "api")
    run = jobs.claim("api", 1)
    assert run is not None
    with pytest.raises(ValueError, match="running"):
        jobs.retry(run.id)
    jobs.finish(run.id, run.attempt, "failed", error="boom")

    again = jobs.retry(run.id)
    assert (again.status, again.attempt, again.error, again.finished_at) == (
        "queued",
        0,
        None,
        None,
    )
    claimed = jobs.claim("api", 1)
    assert claimed is not None and claimed.attempt == 1
    with pytest.raises(KeyError):
        jobs.retry("job_missing")


def test_succeeded_and_canceled_jobs_cannot_be_retried(jobs: JobRepo) -> None:
    jobs.enqueue("s", {}, "cpu")
    run = jobs.claim("cpu", 1)
    assert run is not None
    jobs.finish(run.id, run.attempt, "succeeded")
    with pytest.raises(ValueError):
        jobs.retry(run.id)


def test_requeue_stale_only_touches_running_jobs_with_old_heartbeats(jobs: JobRepo) -> None:
    t0 = utcnow()
    jobs.enqueue("s", {}, "cpu")
    jobs.enqueue("s", {}, "cpu")
    dead = jobs.claim("cpu", 5, now=t0)
    alive = jobs.claim("cpu", 5, now=t0)
    queued = jobs.enqueue("s", {}, "cpu")
    assert dead is not None and alive is not None

    later = t0 + timedelta(seconds=40)
    jobs.heartbeat(alive.id, alive.attempt, now=later)

    assert jobs.requeue_stale(30, now=later) == [dead.id]
    assert jobs.get(dead.id).status == "queued"  # type: ignore[union-attr]
    assert jobs.get(alive.id).status == "running"  # type: ignore[union-attr]
    assert jobs.get(queued.id).status == "queued"  # type: ignore[union-attr]
    assert jobs.requeue_stale(30, now=later) == []  # idempotent

    rerun = jobs.claim("cpu", 5, now=later)
    assert rerun is not None and rerun.id == dead.id and rerun.attempt == 2


def test_requeued_run_cannot_overwrite_its_replacement(jobs: JobRepo) -> None:
    """A worker that stalled and was declared dead must not clobber the new run's state."""
    t0 = utcnow()
    jobs.enqueue("s", {}, "cpu")
    old = jobs.claim("cpu", 1, now=t0)
    assert old is not None
    jobs.requeue_stale(30, now=t0 + timedelta(seconds=60))
    new = jobs.claim("cpu", 1, now=t0 + timedelta(seconds=61))
    assert new is not None and new.attempt == 2

    assert not jobs.finish(old.id, old.attempt, "succeeded")
    assert jobs.heartbeat(old.id, old.attempt) is None
    assert jobs.get(new.id).status == "running"  # type: ignore[union-attr]


def test_stale_job_with_a_cancel_request_is_canceled_not_requeued(jobs: JobRepo) -> None:
    t0 = utcnow()
    jobs.enqueue("s", {}, "cpu")
    run = jobs.claim("cpu", 1, now=t0)
    assert run is not None
    jobs.request_cancel(run.id)
    assert jobs.requeue_stale(30, now=t0 + timedelta(seconds=60)) == []
    assert jobs.get(run.id).status == "canceled"  # type: ignore[union-attr]
