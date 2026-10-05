"""Reading the MovieIndex over HTTP: transcript, shots (with sprite sheets and the proxy),
scenes, story, and the files they point at."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from offscreen.api.app import create_app
from offscreen.config import AppConfig
from offscreen.services.app import AppServices
from offscreen.services.pipeline import Pipeline, Providers
from offscreen.worker import Worker, WorkerSettings

FAST = WorkerSettings(poll_interval_s=0.02, heartbeat_timeout_s=1.0, progress_interval_s=0.0)


@pytest.fixture
def services(cfg: AppConfig, fakes: Providers, movie: Path) -> Iterator[AppServices]:
    with AppServices(cfg.model_copy(update={"media_roots": [movie.parent]}), providers=fakes) as s:
        yield s


@pytest.fixture
def client(services: AppServices) -> Iterator[TestClient]:
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        yield c


def import_movie(client: TestClient, movie: Path) -> str:
    return str(client.post("/api/assets", json={"path": str(movie)}).json()["id"])


def test_nothing_is_readable_before_the_analysis(client: TestClient, movie: Path) -> None:
    asset = import_movie(client, movie)
    for part in ("transcript", "shots", "scenes", "story"):
        r = client.get(f"/api/assets/{asset}/index/{part}")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found", part
        assert "has not been built" in r.json()["error"]["message"]
    r = client.get("/api/assets/ast_missing/index/story")
    assert r.status_code == 404 and "unknown asset" in r.json()["error"]["message"]


def test_the_analysed_index_can_be_read_and_its_files_fetched(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    client.post(f"/api/assets/{asset}/analyze")
    Worker(services.jobs.jobs, services.jobs.execute, services.cfg.data_dir, FAST).drain(
        timeout_s=120
    )

    transcript = client.get(f"/api/assets/{asset}/index/transcript").json()
    assert transcript["asset_id"] == asset and len(transcript["lines"]) == 6
    assert transcript["lines"][0]["text"] == "Where are you going?"

    scenes = client.get(f"/api/assets/{asset}/index/scenes").json()["scenes"]
    assert scenes and scenes[0]["summary"]
    story = client.get(f"/api/assets/{asset}/index/story").json()
    assert story["logline"] and story["acts"]

    shots = client.get(f"/api/assets/{asset}/index/shots").json()
    spans = [(s["start_ms"], s["end_ms"]) for s in shots["shots"]]
    assert spans[:3] == [(0, 8000), (8000, 16000), (16000, 24000)]
    assert spans[3][0] == 24000 and spans[3][1] >= 30000  # the last shot runs to the film's end
    first = shots["shots"][0]
    assert len(first["keyframes"]) == 3 and first["caption"]["caption"].startswith("雪山里")
    assert first["quality"]["sharpness"] >= 0 and first["sprite"] == {
        "sheet": 0,
        "col": 0,
        "row": 0,
    }
    assert shots["shots"][3]["sprite"] == {"sheet": 0, "col": 3, "row": 0}
    assert shots["tile_width"] > 0 and len(shots["sheets"]) == 1

    # everything the view points at is served by the file route
    for rel in (shots["video"], shots["sheets"][0], *first["keyframes"]):
        assert rel.startswith("artifacts/"), rel
        r = client.get(f"/api/files/{rel}")
        assert r.status_code == 200 and len(r.content) > 0, rel
    assert shots["video"].endswith("proxy_540p.mp4")


def test_shots_are_readable_without_captions(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    with Pipeline(services.cfg, services.jobs.providers, db=services.db) as p:
        p.run_stage("analysis.keyframes", asset)
    shots = client.get(f"/api/assets/{asset}/index/shots").json()["shots"]
    assert len(shots) == 4 and all(s["caption"] is None for s in shots)
    assert client.get(f"/api/assets/{asset}/index/scenes").status_code == 404
