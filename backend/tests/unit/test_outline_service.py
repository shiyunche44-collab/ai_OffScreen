"""Outline service and endpoints: read, edit, reset, generate."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from offscreen.api.app import create_app
from offscreen.config import AppConfig
from offscreen.domain.asset import MediaAsset
from offscreen.domain.project import Project, ProjectOptions
from offscreen.domain.script import ScriptOutline
from offscreen.providers.adapters.fake import FakeFaceAnalyzer, FakeLLM, FakeTTS
from offscreen.services.app import AppServices
from offscreen.services.errors import NotFound
from offscreen.services.pipeline import Providers
from offscreen.store.repos import AssetRepo

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "domain"
SCENES = ["sc_040", "sc_041", "sc_042"]


@pytest.fixture
def services(tmp_path: Path) -> Iterator[AppServices]:
    cfg = AppConfig.model_validate(
        {"data_dir": str(tmp_path / "data"), "tts": {"provider": "edge_tts"}}
    )
    fakes = Providers(llm=FakeLLM({}), tts=FakeTTS(), detector=None, faces=FakeFaceAnalyzer())  # type: ignore[arg-type]
    with AppServices(cfg, providers=fakes) as s:
        yield s


@pytest.fixture
def project(services: AppServices) -> Project:
    asset = MediaAsset.model_validate(json.loads((FIXTURES / "asset.json").read_text("utf-8")))
    AssetRepo(services.db).add(asset)
    return services.library.create_project(asset.id, "p", ProjectOptions(style="suspense"))


@pytest.fixture
def generated(
    services: AppServices, project: Project, monkeypatch: pytest.MonkeyPatch
) -> ScriptOutline:
    """Pretend the analysis and the outline stage have run."""
    outline = ScriptOutline.model_validate(
        json.loads((FIXTURES / "outline.json").read_text("utf-8")) | {"asset_id": project.asset_id}
    )
    monkeypatch.setattr(services.outline, "_generated", lambda _p: outline)
    monkeypatch.setattr(services.outline, "_scene_ids", lambda _p: SCENES)
    return outline


def beats(*specs: tuple[str, list[str], int]) -> list[dict[str, Any]]:
    return [{"beat": n, "scene_refs": r, "target_s": s, "focus": f"讲{n}"} for n, r, s in specs]


def test_reading_before_generation_is_not_found(services: AppServices, project: Project) -> None:
    with pytest.raises(NotFound, match="not been generated"):
        services.outline.get(project.id)
    with pytest.raises(NotFound):
        services.outline.get("prj_missing")


def test_generate_queues_the_outline_stage_once(services: AppServices, project: Project) -> None:
    job = services.outline.generate(project.id)
    assert (job.stage, job.lane, job.status) == ("creation.outline", "api", "queued")
    assert job.scope["options"]["style"] == "suspense"
    assert services.outline.generate(project.id).id == job.id  # a second press queues nothing


def test_generated_outline_is_served_as_is(
    services: AppServices, project: Project, generated: ScriptOutline
) -> None:
    view = services.outline.get(project.id)
    assert (view.edited, view.total_s, view.outline) == (False, 60, generated)


def test_an_edit_replaces_the_generated_outline_and_reset_brings_it_back(
    services: AppServices, project: Project, generated: ScriptOutline
) -> None:
    from offscreen.domain.script import OutlineBeat

    edit = [
        OutlineBeat(beat="hook", scene_refs=["sc_041"], target_s=10, focus="新的开场"),
        OutlineBeat(beat="body", scene_refs=["sc_040", "sc_042"], target_s=70),
    ]
    saved = services.outline.save(project.id, edit)
    assert (saved.edited, saved.total_s) == (True, 80)  # the total is shown, not forced
    assert (saved.outline.style, saved.outline.target_duration_s) == ("suspense", 60)

    again = services.outline.get(project.id)
    assert again.edited and [b.beat for b in again.outline.beats] == ["hook", "body"]
    assert (services.cfg.data_dir / "projects" / project.id / "outline.json").is_file()

    back = services.outline.reset(project.id)
    assert (back.edited, back.outline) == (False, generated)
    assert not (services.cfg.data_dir / "projects" / project.id / "outline.json").exists()
    assert services.outline.get(project.id).edited is False


def test_a_second_edit_starts_from_the_first(
    services: AppServices, project: Project, generated: ScriptOutline
) -> None:
    from offscreen.domain.script import OutlineBeat

    one = [OutlineBeat(beat="a", scene_refs=["sc_040"], target_s=60)]
    services.outline.save(project.id, one)
    two = [OutlineBeat(beat="b", scene_refs=["sc_041"], target_s=60)]
    assert [b.beat for b in services.outline.save(project.id, two).outline.beats] == ["b"]


# --- API --------------------------------------------------------------------------------


@pytest.fixture
def client(services: AppServices) -> Iterator[TestClient]:
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        yield c


def test_api_round_trip(client: TestClient, project: Project, generated: ScriptOutline) -> None:
    url = f"/api/projects/{project.id}/outline"
    first = client.get(url).json()
    assert first["edited"] is False and first["total_s"] == 60

    put = client.put(url, json={"beats": beats(("hook", ["sc_040"], 20), ("end", ["sc_042"], 40))})
    assert put.status_code == 200 and put.json()["edited"] is True
    assert [b["focus"] for b in client.get(url).json()["outline"]["beats"]] == ["讲hook", "讲end"]

    reset = client.delete(url)
    assert reset.status_code == 200 and reset.json()["edited"] is False


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (beats(("hook", ["sc_999"], 20)), "sc_999"),
        (beats(("hook", ["sc_040"], 2)), "至少 3 秒"),
        (beats(("hook", ["sc_040"], 10), ("hook", ["sc_041"], 10)), "names must be unique"),
        (beats(("hook", ["sc_040"], 400)), "合计 400 秒"),
        (beats((" ", ["sc_040"], 20)), "must not be empty"),
    ],
)
def test_api_rejects_edits_that_break_the_rules(
    client: TestClient, project: Project, generated: ScriptOutline, bad: list[Any], message: str
) -> None:
    r = client.put(f"/api/projects/{project.id}/outline", json={"beats": bad})
    assert r.status_code == 422
    assert message in r.json()["error"]["message"]
    assert client.get(f"/api/projects/{project.id}/outline").json()["edited"] is False


def test_api_empty_or_malformed_edit_is_422(
    client: TestClient, project: Project, generated: ScriptOutline
) -> None:
    url = f"/api/projects/{project.id}/outline"
    assert client.put(url, json={"beats": []}).status_code == 422
    assert client.put(url, json={}).status_code == 422


def test_api_generate_returns_a_job(client: TestClient, project: Project) -> None:
    r = client.post(f"/api/projects/{project.id}/outline:generate")
    assert r.status_code == 202 and r.json()["stage"] == "creation.outline"
    assert client.get(f"/api/projects/{project.id}/outline").status_code == 404
    assert client.post("/api/projects/prj_missing/outline:generate").status_code == 404
