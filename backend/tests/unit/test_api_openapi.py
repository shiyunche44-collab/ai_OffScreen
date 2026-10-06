"""The committed OpenAPI document is what the code produces, and describes the error model."""

from __future__ import annotations

import json
from pathlib import Path

from offscreen.api.app import create_app

OPENAPI = Path(__file__).resolve().parents[3] / "docs" / "openapi.json"


def spec() -> dict:  # type: ignore[type-arg]
    return create_app().openapi()


def test_committed_openapi_is_up_to_date() -> None:
    assert OPENAPI.exists(), "missing docs/openapi.json; run `make openapi`"
    committed = json.loads(OPENAPI.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(spec())), "OpenAPI drifted; run `make openapi`"


def test_the_documented_routes() -> None:
    paths = set(spec()["paths"])
    assert paths == {
        "/api/assets",
        "/api/assets/browse",
        "/api/assets/{asset_id}",
        "/api/assets/{asset_id}/analyze",
        "/api/assets/{asset_id}/report",
        "/api/assets/{asset_id}/index/transcript",
        "/api/assets/{asset_id}/index/shots",
        "/api/assets/{asset_id}/index/scenes",
        "/api/assets/{asset_id}/index/story",
        "/api/assets/{asset_id}/index/characters",
        "/api/assets/{asset_id}/shots/search",
        "/api/assets/{asset_id}/annotations/cuts",
        "/api/assets/{asset_id}/annotations/cuts/evaluation",
        "/api/assets/{asset_id}/characters:build",
        "/api/assets/{asset_id}/characters/{character_id}",
        "/api/projects",
        "/api/projects/{project_id}",
        "/api/projects/{project_id}/outline",
        "/api/projects/{project_id}/outline:generate",
        "/api/projects/{project_id}/script",
        "/api/projects/{project_id}/script/versions",
        "/api/projects/{project_id}/script/diff",
        "/api/projects/{project_id}/script/segments/{segment_id}:rewrite",
        "/api/projects/{project_id}/script:restore",
        "/api/projects/{project_id}/script:generate",
        "/api/projects/{project_id}/plan:build",
        "/api/projects/{project_id}/render",
        "/api/jobs",
        "/api/jobs/{job_id}",
        "/api/jobs/{job_id}/log",
        "/api/jobs/{job_id}:cancel",
        "/api/jobs/{job_id}:retry",
        "/api/files/{path}",
        "/api/events",
    }


def test_error_responses_are_documented_with_the_shared_model() -> None:
    s = spec()
    assert {"ErrorResponse", "ErrorBody"} <= set(s["components"]["schemas"])
    for path, ops in s["paths"].items():
        if path == "/api/events":  # a stream: it never answers with an error body
            continue
        for method, op in ops.items():
            for code in ("404", "409", "422"):
                ref = op["responses"][code]["content"]["application/json"]["schema"]["$ref"]
                assert ref.endswith("/ErrorResponse"), (path, method, code)


def test_operation_ids_are_unique() -> None:
    """Generated clients name their functions after them."""
    ids = [op["operationId"] for ops in spec()["paths"].values() for op in ops.values()]
    assert len(ids) == len(set(ids))


def test_nothing_answers_outside_the_api_prefix() -> None:
    """ADR-0002: every other path belongs to the front end."""
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    for path in ("/assets", "/jobs", "/projects/prj_1", "/events", "/files/x"):
        assert client.get(path).status_code == 404, path
