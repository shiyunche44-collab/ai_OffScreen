"""`offscreen serve` for real: uvicorn on a socket, the worker thread, real HTTP and a real SSE
stream, and a worker process killed with SIGKILL in the middle of a job."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from offscreen.config import AppConfig
from offscreen.server import Server
from offscreen.services.pipeline import Providers
from offscreen.stages.analysis.ingest import ingest
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo, JobRepo


def wait_for(cond: Callable[[], Any], timeout: float = 60.0, interval: float = 0.05) -> Any:
    end = time.monotonic() + timeout
    while True:
        value = cond()
        if value:
            return value
        assert time.monotonic() < end, "condition not reached in time"
        time.sleep(interval)


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    d = tmp_path / "dist"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html><title>OffScreen</title>", encoding="utf-8")
    return d


def start(cfg: AppConfig, fakes: Providers, dist: Path | None, with_worker: bool = True):  # type: ignore[no-untyped-def]
    server = Server(cfg, port=0, web_dir=dist, with_worker=with_worker, providers=fakes)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert server.started.wait(30), "server did not start"
    client = httpx.Client(base_url=f"http://127.0.0.1:{server.port}", trust_env=False, timeout=30)
    return server, thread, client


@pytest.fixture
def running(
    cfg: AppConfig, fakes: Providers, movie: Path, dist: Path
) -> Iterator[tuple[Server, httpx.Client]]:
    rooted = cfg.model_copy(update={"media_roots": [movie.parent]})
    server, thread, client = start(rooted, fakes, dist)
    yield server, client
    client.close()
    server.request_stop()
    thread.join(30)
    assert not thread.is_alive(), "the server did not stop"


def test_one_process_serves_the_app_and_runs_the_jobs(
    running: tuple[Server, httpx.Client], movie: Path
) -> None:
    _, client = running

    # the front end and the API on one origin
    assert "OffScreen" in client.get("/").text
    assert "OffScreen" in client.get("/projects/prj_anything").text  # client-side route
    assert client.get("/api/jobs").json() == []
    assert client.get("/api/nope").json()["error"]["code"] == "not_found"

    # a real SSE connection, collected in the background
    events: list[tuple[str, Any]] = []
    stop_reading = threading.Event()

    def read_events() -> None:
        with (
            httpx.Client(base_url=client.base_url, trust_env=False, timeout=None) as c,
            c.stream("GET", "/api/events") as r,
        ):
            name = ""
            for line in r.iter_lines():
                if stop_reading.is_set():
                    return
                if line.startswith("event: "):
                    name = line.removeprefix("event: ")
                elif line.startswith("data: "):
                    events.append((name, json.loads(line.removeprefix("data: "))))

    reader = threading.Thread(target=read_events, daemon=True)
    reader.start()
    wait_for(lambda: events)  # the snapshot arrives first
    assert events[0] == ("snapshot", {"jobs": []})

    asset = client.post("/api/assets", json={"path": str(movie)}).json()
    project = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()
    job = client.post(f"/api/projects/{project['id']}/render").json()
    assert job["status"] in ("queued", "running")

    done = wait_for(
        lambda: (j := client.get(f"/api/jobs/{job['id']}").json())["status"] == "succeeded" and j
    )
    assert done["progress"] == 1.0

    # the SSE stream told the browser about it, in order, ending in success
    wait_for(
        lambda: any(
            n == "job" and d["id"] == job["id"] and d["status"] == "succeeded" for n, d in events
        )
    )
    seen = [d["status"] for n, d in events if n == "job" and d["id"] == job["id"]]
    assert seen[0] in ("queued", "running") and seen[-1] == "succeeded"
    progress = [
        d["progress"]
        for n, d in events
        if n == "job" and d["id"] == job["id"] and d["status"] == "running"
    ]
    assert progress == sorted(progress)

    # and the film plays: project detail names the file, the file service streams ranges
    detail = client.get(f"/api/projects/{project['id']}").json()
    assert detail["video"].endswith("/final.mp4")
    head = client.get(f"/api/files/{detail['video']}", headers={"Range": "bytes=0-11"})
    assert head.status_code == 206 and len(head.content) == 12
    assert head.headers["content-range"].startswith("bytes 0-11/")
    stop_reading.set()


def test_without_the_worker_jobs_wait_in_the_queue(
    cfg: AppConfig, fakes: Providers, movie: Path
) -> None:
    rooted = cfg.model_copy(update={"media_roots": [movie.parent]})
    server, thread, client = start(rooted, fakes, None, with_worker=False)
    try:
        asset = client.post("/api/assets", json={"path": str(movie)}).json()
        job = client.post(f"/api/assets/{asset['id']}/analyze").json()
        time.sleep(1.0)
        assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "queued"
        assert client.get("/").status_code == 404  # no front end given
    finally:
        client.close()
        server.request_stop()
        thread.join(30)


def test_stopping_the_server_puts_unfinished_jobs_back_in_the_queue(
    cfg: AppConfig, fakes: Providers, movie: Path
) -> None:
    started = threading.Event()

    class Blocking:
        id = "blocking@1"

        def detect(
            self, video: Path, *, on_progress: Any = None, should_cancel: Any = None
        ) -> list[Any]:
            started.set()
            while not should_cancel():
                time.sleep(0.01)
            from offscreen.providers.ports import DetectionCanceled

            raise DetectionCanceled("stopped")

    fakes.detector = Blocking()  # type: ignore[assignment]
    rooted = cfg.model_copy(update={"media_roots": [movie.parent]})
    server, thread, client = start(rooted, fakes, None)
    asset = client.post("/api/assets", json={"path": str(movie)}).json()
    job = client.post(f"/api/assets/{asset['id']}/analyze").json()
    assert started.wait(60)
    client.close()
    server.request_stop()
    thread.join(30)
    assert not thread.is_alive()

    db = Database(cfg.data_dir / "offscreen.db")
    try:
        back = JobRepo(db).get(job["id"])
    finally:
        db.close()
    assert back is not None and back.status == "queued"  # not failed, not canceled: resumable
    assert back.message == "requeued: worker stopped"


# --- a worker process killed in the middle of a job -------------------------------------


def run_cli(*args: str) -> subprocess.Popen[bytes]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return subprocess.Popen(
        [sys.executable, "-m", "offscreen.cli", *args],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_a_worker_killed_mid_job_is_replaced_and_the_job_completes(
    cfg: AppConfig, movie: Path, tmp_path: Path
) -> None:
    # A worker process builds the real adapters, so the config must be one they accept.
    raw = json.loads(cfg.model_dump_json())
    raw["providers"]["minimax"] = raw["providers"]["p"]
    raw["tts"] = {"provider": "minimax", "default_voice": "voice-x"}
    raw["worker"] = {"heartbeat_timeout_s": 3.0}
    conf = AppConfig.model_validate(raw)
    conf_path = tmp_path / "config.yaml"
    conf_path.write_text(yaml.safe_dump(json.loads(conf.model_dump_json())), encoding="utf-8")

    db = Database(conf.data_dir / "offscreen.db")
    try:
        jobs, assets = JobRepo(db), AssetRepo(db)
        asset_id = ingest(movie, assets).asset.id
        job = jobs.enqueue("analysis.proxy", {"asset_id": asset_id, "options": {}}, "cpu")

        first = run_cli("worker", "--config", str(conf_path))
        try:
            wait_for(lambda: jobs.get(job.id).status == "running", timeout=60, interval=0.01)  # type: ignore[union-attr]
            first.send_signal(signal.SIGKILL)  # no chance to clean up
            first.wait(10)
        finally:
            if first.poll() is None:
                first.kill()
        stuck = jobs.get(job.id)
        assert stuck is not None and stuck.status == "running" and stuck.attempt == 1  # orphaned

        second = run_cli("worker", "--config", str(conf_path))
        try:
            done = wait_for(
                lambda: (
                    (j := jobs.get(job.id)) is not None
                    and j.status in ("succeeded", "failed")
                    and j
                ),
                timeout=90,
            )
            assert done.status == "succeeded", done.error
            assert done.attempt == 2  # it ran again after the heartbeat went stale
        finally:
            second.send_signal(signal.SIGTERM)
            try:
                second.wait(20)
            except subprocess.TimeoutExpired:
                second.kill()
    finally:
        db.close()
    assert list((conf.data_dir / "artifacts" / "analysis.proxy").glob("*/proxy.mp4")) or list(
        (conf.data_dir / "artifacts" / "analysis.proxy").iterdir()
    )
