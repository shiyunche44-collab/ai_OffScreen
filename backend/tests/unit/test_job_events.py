"""The live job feed: change detection, SSE framing, and the stream over HTTP."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from offscreen.api.app import create_app
from offscreen.api.sse import EventSettings, format_event, job_events
from offscreen.config import AppConfig
from offscreen.services.app import AppServices
from offscreen.services.events import JobWatcher
from offscreen.store.models import utcnow
from offscreen.store.repos import JobRepo


@pytest.fixture
def services(tmp_path: Path) -> Iterator[AppServices]:
    cfg = AppConfig.model_validate(
        {"data_dir": str(tmp_path / "data"), "tts": {"provider": "edge_tts"}}
    )
    with AppServices(cfg) as s:
        yield s


@pytest.fixture
def jobs(services: AppServices) -> JobRepo:
    return services.jobs.jobs


def ids(items: list[Any]) -> list[str]:
    return [j.id for j in items]


# --- JobWatcher -------------------------------------------------------------------------


def test_snapshot_has_active_jobs_and_recently_finished_ones_only(jobs: JobRepo) -> None:
    old = jobs.enqueue("s", {}, "cpu")
    run = jobs.claim("cpu", 1, now=utcnow() - timedelta(hours=2))
    assert run is not None
    jobs.finish(old.id, run.attempt, "succeeded", now=utcnow() - timedelta(hours=1))

    fresh = jobs.enqueue("s", {}, "cpu")
    done = jobs.enqueue("s", {}, "cpu")
    r2 = jobs.claim("cpu", 1)
    assert r2 is not None and r2.id == fresh.id
    r3 = jobs.claim("cpu", 2)
    assert r3 is not None and r3.id == done.id
    jobs.finish(done.id, r3.attempt, "failed", error="boom")
    queued = jobs.enqueue("s", {}, "api")

    snap = JobWatcher(jobs).snapshot()
    assert ids(snap) == [fresh.id, done.id, queued.id]  # no hour-old job
    assert {j.id: j.status for j in snap} == {
        fresh.id: "running",
        done.id: "failed",
        queued.id: "queued",
    }


def test_poll_reports_only_what_changed(jobs: JobRepo) -> None:
    a = jobs.enqueue("s", {}, "cpu")
    w = JobWatcher(jobs)
    w.snapshot()
    assert w.poll() == []  # nothing happened

    b = jobs.enqueue("s", {}, "cpu")  # new
    assert ids(w.poll()) == [b.id]
    assert w.poll() == []

    run = jobs.claim("cpu", 2)  # a starts
    assert run is not None and run.id == a.id
    assert [(j.id, j.status) for j in w.poll()] == [(a.id, "running")]

    jobs.report_progress(a.id, run.attempt, 0.4, "halfway")
    (changed,) = w.poll()
    assert (changed.id, changed.progress, changed.message) == (a.id, 0.4, "halfway")
    jobs.report_progress(a.id, run.attempt, 0.4, "halfway")  # same values again
    assert w.poll() == []


def test_a_finished_job_is_reported_once_with_its_final_state(jobs: JobRepo) -> None:
    job = jobs.enqueue("s", {}, "cpu")
    run = jobs.claim("cpu", 1)
    assert run is not None
    w = JobWatcher(jobs)
    w.snapshot()
    jobs.finish(job.id, run.attempt, "succeeded", message="done")
    (final,) = w.poll()
    assert (final.id, final.status, final.progress, final.message) == (
        job.id,
        "succeeded",
        1.0,
        "done",
    )
    assert w.poll() == []


def test_failure_cancel_and_manual_retry_are_all_seen(jobs: JobRepo) -> None:
    w = JobWatcher(jobs)
    w.snapshot()
    failing = jobs.enqueue("s", {}, "cpu")
    w.poll()
    run = jobs.claim("cpu", 1)
    assert run is not None
    jobs.finish(failing.id, run.attempt, "failed", error="boom")
    states = [(j.status, j.error) for j in w.poll()]
    assert states == [("failed", "boom")]

    jobs.retry(failing.id)  # back to queued
    assert [(j.id, j.status) for j in w.poll()] == [(failing.id, "queued")]

    jobs.request_cancel(failing.id)
    assert [(j.id, j.status) for j in w.poll()] == [(failing.id, "canceled")]


def test_a_retried_old_failure_is_picked_up_even_if_it_is_not_in_the_snapshot(
    jobs: JobRepo,
) -> None:
    job = jobs.enqueue("s", {}, "cpu")
    run = jobs.claim("cpu", 1, now=utcnow() - timedelta(hours=3))
    assert run is not None
    jobs.finish(job.id, run.attempt, "failed", error="x", now=utcnow() - timedelta(hours=2))
    w = JobWatcher(jobs)
    assert w.snapshot() == []
    jobs.retry(job.id)
    assert ids(w.poll()) == [job.id]


def test_watchers_are_independent(jobs: JobRepo) -> None:
    w1, w2 = JobWatcher(jobs), JobWatcher(jobs)
    w1.snapshot()
    w2.snapshot()
    job = jobs.enqueue("s", {}, "cpu")
    assert ids(w1.poll()) == [job.id] and ids(w2.poll()) == [job.id]


# --- framing ----------------------------------------------------------------------------


def test_format_event() -> None:
    assert format_event("job", '{"a":1}') == 'event: job\ndata: {"a":1}\n\n'
    with pytest.raises(ValueError):
        format_event("job", "two\nlines")


# --- the feed loop ----------------------------------------------------------------------


FAST = EventSettings(poll_s=0.01, keepalive_s=0.05, max_lifetime_s=0.4, retry_ms=1234)


async def collect(
    watcher: JobWatcher, settings: EventSettings = FAST, disconnect_after: int | None = None
) -> list[str]:
    calls = 0

    async def is_disconnected() -> bool:
        nonlocal calls
        calls += 1
        return disconnect_after is not None and calls > disconnect_after

    return [chunk async for chunk in job_events(watcher, is_disconnected, settings)]


def parse(chunks: list[str]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for c in chunks:
        if c.startswith("event: "):
            head, data = c.split("\n", 2)[:2]
            out.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_the_feed_starts_with_retry_and_snapshot_then_ends_by_itself(jobs: JobRepo) -> None:
    job = jobs.enqueue("s", {}, "cpu")
    chunks = asyncio.run(collect(JobWatcher(jobs)))
    assert chunks[0] == "retry: 1234\n\n"
    events = parse(chunks)
    assert events[0][0] == "snapshot"
    assert [j["id"] for j in events[0][1]["jobs"]] == [job.id]
    assert any(c.startswith(":") for c in chunks[2:])  # idle: keep-alive comments


def test_changes_arrive_as_job_events_while_the_feed_runs(jobs: JobRepo) -> None:
    async def scenario() -> list[str]:
        async def mutate() -> None:
            await asyncio.sleep(0.1)
            job = jobs.enqueue("s", {}, "cpu")
            await asyncio.sleep(0.1)
            run = jobs.claim("cpu", 1)
            assert run is not None and run.id == job.id
            await asyncio.sleep(0.1)  # polls in between see "running" before the progress
            jobs.report_progress(job.id, run.attempt, 0.5, "half")

        task = asyncio.create_task(mutate())
        chunks = await collect(JobWatcher(jobs), EventSettings(poll_s=0.01, max_lifetime_s=0.6))
        await task
        return chunks

    events = parse(asyncio.run(scenario()))
    assert events[0] == ("snapshot", {"jobs": []})
    seen = [(e[1]["status"], e[1]["progress"]) for e in events[1:] if e[0] == "job"]
    assert seen == [("queued", 0.0), ("running", 0.0), ("running", 0.5)]


def test_the_feed_stops_when_the_client_goes_away(jobs: JobRepo) -> None:
    long = EventSettings(poll_s=0.01, max_lifetime_s=60.0)
    chunks = asyncio.run(asyncio.wait_for(collect(JobWatcher(jobs), long, disconnect_after=3), 5))
    assert parse(chunks)[0][0] == "snapshot"  # and it returned instead of running for a minute


# --- over HTTP --------------------------------------------------------------------------


def test_events_endpoint_streams_snapshot_and_changes(services: AppServices) -> None:
    jobs = services.jobs.jobs
    existing = jobs.enqueue("analysis.story", {"asset_id": "ast_1"}, "api")

    def mutate() -> None:
        jobs.enqueue("creation.script", {"asset_id": "ast_1"}, "api")
        run = jobs.claim("api", 4)
        assert run is not None
        jobs.report_progress(run.id, run.attempt, 0.3, "writing")  # the older job starts first

    timer = threading.Timer(0.3, mutate)
    settings = EventSettings(poll_s=0.05, keepalive_s=5, max_lifetime_s=1.2, retry_ms=500)
    timer.start()
    with TestClient(create_app(services, events_settings=settings)) as client:
        r = client.get("/api/events")
    timer.join()

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    assert r.text.startswith("retry: 500\n\n")
    events = parse(r.text.split("\n\n"))
    kind, body = events[0]
    assert kind == "snapshot" and [j["id"] for j in body["jobs"]] == [existing.id]
    changes = [e[1] for e in events[1:] if e[0] == "job"]
    assert {c["stage"] for c in changes} == {"analysis.story", "creation.script"}
    assert any(c["message"] == "writing" for c in changes)


def test_events_endpoint_is_documented_as_event_stream() -> None:
    spec = create_app().openapi()
    ok = spec["paths"]["/api/events"]["get"]["responses"]["200"]
    assert "text/event-stream" in ok["content"] and "snapshot" in ok["description"]


def test_a_job_that_starts_and_finishes_between_two_polls_is_still_reported(jobs: JobRepo) -> None:
    w = JobWatcher(jobs)
    w.snapshot()

    quick = jobs.enqueue("s", {}, "cpu")  # a cache hit, a short rewrite
    run = jobs.claim("cpu", 1)
    assert run is not None
    jobs.finish(quick.id, run.attempt, "succeeded")

    reported = w.poll()
    assert ids(reported) == [quick.id] and reported[0].status == "succeeded"
    assert w.poll() == []  # and only once


def test_a_job_seen_running_is_reported_once_when_it_finishes(jobs: JobRepo) -> None:
    job = jobs.enqueue("s", {}, "cpu")
    w = JobWatcher(jobs)
    w.snapshot()
    run = jobs.claim("cpu", 1)
    assert run is not None
    assert ids(w.poll()) == [job.id]  # running

    jobs.finish(job.id, run.attempt, "succeeded")
    done = w.poll()
    assert [(j.id, j.status) for j in done] == [(job.id, "succeeded")]  # not twice
    assert w.poll() == []


def test_jobs_finished_before_the_snapshot_are_not_replayed(jobs: JobRepo) -> None:
    old = jobs.enqueue("s", {}, "cpu")
    run = jobs.claim("cpu", 1)
    assert run is not None
    jobs.finish(old.id, run.attempt, "succeeded")
    w = JobWatcher(jobs)
    assert ids(w.snapshot()) == [old.id]
    assert w.poll() == []
