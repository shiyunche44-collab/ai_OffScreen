"""The HTTP API over real services (fake LLM / TTS / shot detection, real ffmpeg and stores).
A Worker runs the queued jobs, as the server process will."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from offscreen.api.app import create_app
from offscreen.config import AppConfig
from offscreen.services.app import AppServices
from offscreen.services.pipeline import Providers
from offscreen.worker import Worker, WorkerSettings

FAST = WorkerSettings(poll_interval_s=0.02, heartbeat_timeout_s=1.0, progress_interval_s=0.0)


@pytest.fixture
def services(cfg: AppConfig, fakes: Providers, movie: Path) -> Iterator[AppServices]:
    rooted = cfg.model_copy(update={"media_roots": [movie.parent]})
    with AppServices(rooted, providers=fakes) as s:
        yield s


@pytest.fixture
def client(services: AppServices) -> Iterator[TestClient]:
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        yield c


def drain(services: AppServices) -> None:
    Worker(
        services.db and services.jobs.jobs, services.jobs.execute, services.cfg.data_dir, FAST
    ).drain(timeout_s=120)


def err(r: Any, status: int, code: str) -> dict[str, Any]:
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and body["error"]["code"] == code, body
    assert body["error"]["message"]
    return body["error"]  # type: ignore[no-any-return]


def import_movie(client: TestClient, movie: Path) -> dict[str, Any]:
    r = client.post("/api/assets", json={"path": str(movie)})
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


# --- assets -----------------------------------------------------------------------------


def test_import_list_and_get_an_asset(client: TestClient, movie: Path) -> None:
    asset = import_movie(client, movie)
    assert asset["id"].startswith("ast_") and asset["title"]
    again = import_movie(client, movie)
    assert again["id"] == asset["id"]  # same file, same asset
    assert [a["id"] for a in client.get("/api/assets").json()] == [asset["id"]]

    detail = client.get(f"/api/assets/{asset['id']}").json()
    assert detail["asset"]["id"] == asset["id"]
    assert [s["stage"] for s in detail["stages"]] == [
        "analysis.proxy",
        "analysis.transcript",
        "analysis.shots",
        "analysis.story",
    ]
    assert not any(s["cached"] for s in detail["stages"])


def test_import_is_limited_to_the_media_roots(
    client: TestClient, movie: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere.mp4"
    outside.write_bytes(movie.read_bytes())
    err(client.post("/api/assets", json={"path": str(outside)}), 422, "invalid_input")
    sneaky = movie.parent / ".." / "elsewhere.mp4"
    err(client.post("/api/assets", json={"path": str(sneaky)}), 422, "invalid_input")
    err(
        client.post("/api/assets", json={"path": str(movie.parent)}), 422, "invalid_input"
    )  # a folder
    err(
        client.post("/api/assets", json={"path": str(movie.parent / "nope.mp4")}),
        422,
        "invalid_input",
    )
    assert client.get("/api/assets").json() == []


def test_a_file_that_is_not_a_video_is_refused(client: TestClient, movie: Path) -> None:
    junk = movie.parent / "notes.mp4"
    junk.write_text("this is not a movie")
    err(client.post("/api/assets", json={"path": str(junk)}), 422, "invalid_input")


def test_import_without_media_roots_explains_what_to_configure(
    cfg: AppConfig, fakes: Providers, movie: Path
) -> None:
    with AppServices(cfg, providers=fakes) as s, TestClient(create_app(s)) as c:
        e = err(c.post("/api/assets", json={"path": str(movie)}), 422, "invalid_input")
        assert "media_roots" in e["message"]


def test_unknown_asset_is_a_404_with_the_error_shape(client: TestClient) -> None:
    err(client.get("/api/assets/ast_missing"), 404, "not_found")
    err(client.post("/api/assets/ast_missing/analyze"), 404, "not_found")


def test_request_validation_errors_list_the_fields(client: TestClient) -> None:
    e = err(client.post("/api/assets", json={"title": "x"}), 422, "validation_error")
    assert e["details"] and e["details"][0]["loc"] == ["body", "path"]
    err(client.post("/api/assets", json={"path": ""}), 422, "validation_error")


def test_analyze_runs_as_a_job_and_updates_the_asset_status(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    r = client.post(f"/api/assets/{asset['id']}/analyze")
    assert r.status_code == 202
    job = r.json()
    assert (job["status"], job["stage"], job["lane"]) == ("queued", "analysis.story", "api")
    assert (
        client.post(f"/api/assets/{asset['id']}/analyze").json()["id"] == job["id"]
    )  # no duplicate

    drain(services)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert (done["status"], done["progress"]) == ("succeeded", 1.0)
    log = client.get(f"/api/jobs/{job['id']}/log")
    assert log.status_code == 200 and log.headers["content-type"].startswith("text/plain")
    assert "start analysis.story" in log.text

    stages = client.get(f"/api/assets/{asset['id']}").json()["stages"]
    assert all(s["cached"] for s in stages)


# --- projects ---------------------------------------------------------------------------


def test_create_project_defaults_and_validation(client: TestClient, movie: Path) -> None:
    asset = import_movie(client, movie)
    r = client.post("/api/projects", json={"asset_id": asset["id"]})
    assert r.status_code == 201
    project = r.json()
    assert project["id"].startswith("prj_") and project["name"] == asset["title"]
    assert project["options"] == {
        "minutes": 3.0,
        "voice": None,
        "style": "neutral",
        "spoil_ending": True,
    }

    named = client.post(
        "/api/projects",
        json={
            "asset_id": asset["id"],
            "name": "短版",
            "options": {"minutes": 0.25, "style": "funny"},
        },
    ).json()
    assert named["name"] == "短版" and named["options"]["minutes"] == 0.25
    assert [p["id"] for p in client.get("/api/projects").json()] == [named["id"], project["id"]]

    err(client.post("/api/projects", json={"asset_id": "ast_missing"}), 404, "not_found")
    err(client.post("/api/projects", json={"asset_id": "nonsense"}), 422, "validation_error")
    bad = {"asset_id": asset["id"], "options": {"minutes": 0}}
    err(client.post("/api/projects", json=bad), 422, "validation_error")
    err(client.get("/api/projects/prj_missing"), 404, "not_found")


def test_a_project_goes_from_nothing_to_a_playable_video(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset = import_movie(client, movie)
    project = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.25}}
    ).json()
    pid = project["id"]
    before = client.get(f"/api/projects/{pid}").json()
    assert before["video"] is None and not any(s["cached"] for s in before["stages"])

    jobs = []
    for step in ("script:generate", "plan:build", "render"):
        r = client.post(f"/api/projects/{pid}/{step}")
        assert r.status_code == 202, r.text
        jobs.append(r.json())
    assert [j["stage"] for j in jobs] == ["creation.script", "creation.plan", "output.render"]
    drain(services)
    assert all(client.get(f"/api/jobs/{j['id']}").json()["status"] == "succeeded" for j in jobs)

    after = client.get(f"/api/projects/{pid}").json()
    assert all(s["cached"] for s in after["stages"])
    assert after["video"].endswith("/final.mp4")
    assert (services.cfg.data_dir / after["video"]).is_file()

    other = client.post(
        "/api/projects", json={"asset_id": asset["id"], "options": {"minutes": 0.3}}
    ).json()
    cached = {
        s["stage"]
        for s in client.get(f"/api/projects/{other['id']}").json()["stages"]
        if s["cached"]
    }
    assert cached == {  # same film, other length: only the analysis carries over
        "analysis.proxy",
        "analysis.shots",
        "analysis.transcript",
        "analysis.story",
    }
    assert client.get(f"/api/projects/{other['id']}").json()["video"] is None


# --- jobs -------------------------------------------------------------------------------


def test_job_control_over_http(client: TestClient, services: AppServices, movie: Path) -> None:
    asset = import_movie(client, movie)
    job = client.post(f"/api/assets/{asset['id']}/analyze").json()

    assert [j["id"] for j in client.get("/api/jobs", params={"status": "queued"}).json()] == [
        job["id"]
    ]
    assert client.get("/api/jobs", params={"lane": "gpu"}).json() == []
    err(client.get("/api/jobs", params={"status": "bogus"}), 422, "validation_error")

    err(client.post(f"/api/jobs/{job['id']}:retry"), 409, "conflict")  # not failed
    canceled = client.post(f"/api/jobs/{job['id']}:cancel")
    assert canceled.status_code == 200 and canceled.json()["status"] == "canceled"
    assert client.post(f"/api/jobs/{job['id']}:cancel").json()["status"] == "canceled"  # idempotent
    drain(services)  # nothing left to run
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "canceled"
    assert client.get(f"/api/jobs/{job['id']}/log").text == ""

    err(client.get("/api/jobs/job_missing"), 404, "not_found")
    err(client.get("/api/jobs/job_missing/log"), 404, "not_found")
    err(client.post("/api/jobs/job_missing:cancel"), 404, "not_found")
    err(client.post("/api/jobs/job_missing:retry"), 404, "not_found")


# --- error model ------------------------------------------------------------------------


def test_unknown_route_and_wrong_method_use_the_error_shape(client: TestClient) -> None:
    err(client.get("/api/nothing-here"), 404, "not_found")
    err(client.delete("/api/assets"), 405, "method_not_allowed")


def test_unexpected_failures_are_500_without_leaking_details(
    cfg: AppConfig, fakes: Providers, services: AppServices
) -> None:
    def boom(**_: Any) -> Any:
        raise RuntimeError("secret internal detail /home/x/key")

    services.jobs.list = boom  # type: ignore[method-assign,assignment]
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        e = err(c.get("/api/jobs"), 500, "internal")
    assert e["message"] == "internal error" and "secret" not in str(e)
