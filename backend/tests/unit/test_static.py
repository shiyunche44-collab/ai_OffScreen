"""Serving the built front end next to the API."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from offscreen.api.app import create_app
from offscreen.api.static import mount_frontend
from offscreen.config import AppConfig
from offscreen.services.app import AppServices

INDEX = "<!doctype html><title>app</title><div id=root></div>"


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(INDEX, encoding="utf-8")
    (d / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (d / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside the build", encoding="utf-8")
    return d


@pytest.fixture
def client(dist: Path, tmp_path: Path) -> Iterator[TestClient]:
    cfg = AppConfig.model_validate(
        {"data_dir": str(tmp_path / "data"), "tts": {"provider": "edge_tts"}}
    )
    with AppServices(cfg) as s, TestClient(create_app(s, web_dir=dist)) as c:
        yield c


def test_the_root_and_every_client_route_get_index_html(client: TestClient) -> None:
    for path in ("/", "/library", "/jobs", "/projects/prj_01ABC", "/some/deep/route"):
        r = client.get(path)
        assert r.status_code == 200 and r.text == INDEX, path
        assert r.headers["content-type"].startswith("text/html")
        assert r.headers["cache-control"] == "no-cache"


def test_build_files_are_served_with_the_right_caching(client: TestClient) -> None:
    js = client.get("/assets/index-abc123.js")
    assert js.status_code == 200 and js.text == "console.log(1)"
    assert "immutable" in js.headers["cache-control"]  # hashed name: never changes
    icon = client.get("/favicon.svg")
    assert icon.status_code == 200 and icon.headers["cache-control"] == "no-cache"


def test_a_missing_build_file_falls_back_to_the_app(client: TestClient) -> None:
    assert client.get("/assets/nope.js").text == INDEX  # the router shows its own 404 page


def test_the_api_still_answers_and_unknown_api_paths_are_json_errors(client: TestClient) -> None:
    assert client.get("/api/jobs").json() == []
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"  # not the HTML page
    assert client.get("/api").status_code == 404
    assert client.get("/openapi.json").json()["info"]["title"] == "AI OffScreen"


@pytest.mark.parametrize(
    "path",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/assets/../../secret.txt",
        "/..%2fsecret.txt",
        "//etc/passwd",
        "/%00",
    ],
)
def test_nothing_outside_the_build_directory_is_served(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert "outside the build" not in r.text and "root:" not in r.text
    assert r.text == INDEX or r.status_code in (404, 422)


def test_writes_to_client_routes_are_refused(client: TestClient) -> None:
    assert client.post("/library").status_code == 405


def test_mounting_needs_a_built_front_end(tmp_path: Path) -> None:
    app = create_app()
    with pytest.raises(FileNotFoundError, match="npm run build"):
        mount_frontend(app, tmp_path / "empty")


def test_without_a_front_end_the_root_is_not_found(tmp_path: Path) -> None:
    cfg = AppConfig.model_validate(
        {"data_dir": str(tmp_path / "data"), "tts": {"provider": "edge_tts"}}
    )
    with AppServices(cfg) as s, TestClient(create_app(s)) as c:
        assert c.get("/").status_code == 404
