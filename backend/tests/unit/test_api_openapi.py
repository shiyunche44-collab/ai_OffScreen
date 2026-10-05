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


def test_every_documented_route_belongs_to_assets_projects_or_jobs() -> None:
    paths = set(spec()["paths"])
    assert paths == {
        "/assets",
        "/assets/{asset_id}",
        "/assets/{asset_id}/analyze",
        "/projects",
        "/projects/{project_id}",
        "/projects/{project_id}/script:generate",
        "/projects/{project_id}/plan:build",
        "/projects/{project_id}/render",
        "/jobs",
        "/jobs/{job_id}",
        "/jobs/{job_id}/log",
        "/jobs/{job_id}:cancel",
        "/jobs/{job_id}:retry",
    }


def test_error_responses_are_documented_with_the_shared_model() -> None:
    s = spec()
    assert {"ErrorResponse", "ErrorBody"} <= set(s["components"]["schemas"])
    for path, ops in s["paths"].items():
        for method, op in ops.items():
            for code in ("404", "409", "422"):
                ref = op["responses"][code]["content"]["application/json"]["schema"]["$ref"]
                assert ref.endswith("/ErrorResponse"), (path, method, code)
