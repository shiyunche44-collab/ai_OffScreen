"""The analysis report: time per stage from `stage_runs`, model usage from `llm_calls`, counts
from the artifacts. Real services and stores, fake models."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from offscreen.api.app import create_app
from offscreen.cli import app
from offscreen.config import AppConfig
from offscreen.domain.index import Scenes
from offscreen.providers.adapters.fake import FakeTTS
from offscreen.services.app import AppServices
from offscreen.services.llm import call_recorder
from offscreen.services.pipeline import Pipeline, Providers
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.store.db import Database
from offscreen.worker import Worker, WorkerSettings

from .conftest import FixedShots, scripted_llm

FAST = WorkerSettings(poll_interval_s=0.02, heartbeat_timeout_s=1.0, progress_interval_s=0.0)
CHAIN = [
    "analysis.proxy",
    "analysis.shots",
    "analysis.keyframes",
    "analysis.transcript",
    "analysis.captions",
    "analysis.scenes",
    "analysis.story",
]


@pytest.fixture
def recording(cfg: AppConfig) -> Iterator[Providers]:
    """Fake models whose calls are written to the same database the services use."""
    db = Database(cfg.data_dir / "offscreen.db")
    try:
        yield Providers(
            llm=scripted_llm(call_recorder(cfg, db)),
            tts=FakeTTS(chars_per_s=4.5),
            detector=FixedShots(),
        )
    finally:
        db.close()


@pytest.fixture
def services(cfg: AppConfig, recording: Providers, movie: Path) -> Iterator[AppServices]:
    with AppServices(
        cfg.model_copy(update={"media_roots": [movie.parent]}), providers=recording
    ) as s:
        yield s


@pytest.fixture
def client(services: AppServices) -> Iterator[TestClient]:
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        yield c


def analyze_with_a_worker(client: TestClient, services: AppServices, movie: Path) -> str:
    asset_id = client.post("/api/assets", json={"path": str(movie)}).json()["id"]
    client.post(f"/api/assets/{asset_id}/analyze")
    Worker(services.jobs.jobs, services.jobs.execute, services.cfg.data_dir, FAST).drain(
        timeout_s=120
    )
    return str(asset_id)


def test_report_of_an_analysis_run_by_the_worker(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset_id = analyze_with_a_worker(client, services, movie)
    r = client.get(f"/api/assets/{asset_id}/report")
    assert r.status_code == 200, r.text
    report = r.json()

    assert report["asset"]["id"] == asset_id and report["complete"] is True
    assert [s["stage"] for s in report["stages"]] == CHAIN
    for s in report["stages"]:
        assert (s["cached"], s["runs"], s["failed_runs"]) == (True, 1, 0), s
        assert s["last_run_ms"] is not None and s["total_ms"] == s["last_run_ms"]
    assert report["total_ms"] == sum(s["total_ms"] for s in report["stages"])
    with Pipeline(services.cfg, services.jobs.providers, db=services.db) as p:
        scenes = p.peek("analysis.scenes", asset_id)
    assert scenes is not None
    assert report["counts"] == {
        "shots": 4,
        "transcript_lines": 6,
        "scenes": len(scenes.read_model(SCENES_FILE, Scenes).scenes),
        "characters": None,
    }

    by_stage = {s["stage"]: s["llm"] for s in report["stages"]}
    assert by_stage["analysis.captions"]["calls"] >= 1
    assert by_stage["analysis.scenes"]["calls"] >= 2  # boundaries, then summaries
    assert by_stage["analysis.story"]["calls"] == 2  # acts, then synthesis
    for quiet in ("analysis.proxy", "analysis.shots", "analysis.keyframes", "analysis.transcript"):
        assert by_stage[quiet]["calls"] == 0
    assert report["llm"]["calls"] == sum(u["calls"] for u in by_stage.values())

    # the worker's job id reaches the calls and the runs
    job = client.get("/api/jobs").json()[0]
    assert {c.job_id for c in services.report.calls.list(asset_id=asset_id)} == {job["id"]}
    assert {x.job_id for x in services.report.runs.list(asset_id=asset_id)} == {job["id"]}


def test_cached_reruns_add_no_runs_to_the_report(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset_id = analyze_with_a_worker(client, services, movie)
    before = client.get(f"/api/assets/{asset_id}/report").json()

    client.post(f"/api/assets/{asset_id}/analyze")
    Worker(services.jobs.jobs, services.jobs.execute, services.cfg.data_dir, FAST).drain(
        timeout_s=120
    )
    assert client.get(f"/api/assets/{asset_id}/report").json() == before  # all cache hits

    with Pipeline(services.cfg, services.jobs.providers, db=services.db) as p:
        p.run_stage("analysis.story", asset_id)  # still cached: no new rows either
    assert client.get(f"/api/assets/{asset_id}/report").json() == before


def test_report_before_and_partway_through_the_analysis(
    client: TestClient, services: AppServices, movie: Path
) -> None:
    asset_id = client.post("/api/assets", json={"path": str(movie)}).json()["id"]
    empty = client.get(f"/api/assets/{asset_id}/report").json()
    assert empty["complete"] is False and empty["total_ms"] == 0
    assert empty["counts"] == {
        "shots": None,
        "transcript_lines": None,
        "scenes": None,
        "characters": None,
    }
    assert all(s["runs"] == 0 and not s["cached"] for s in empty["stages"])

    with Pipeline(services.cfg, services.jobs.providers, db=services.db) as p:
        p.run_stage("analysis.shots", asset_id)
    part = client.get(f"/api/assets/{asset_id}/report").json()
    assert part["complete"] is False and part["counts"]["shots"] == 4
    assert part["counts"]["scenes"] is None
    built = [s["stage"] for s in part["stages"] if s["cached"]]
    assert built == ["analysis.proxy", "analysis.shots"]


def test_a_failed_run_is_counted_with_its_model_usage(
    cfg: AppConfig, movie: Path, tmp_path: Path
) -> None:
    from offscreen.providers.ports import LLMError

    db = Database(cfg.data_dir / "offscreen.db")
    llm = scripted_llm(call_recorder(cfg, db))
    real = llm._fns["story"]  # type: ignore[attr-defined]
    state = {"fail": True}

    def flaky(task: str, m: Any, schema: Any) -> Any:
        if state["fail"] and "各幕：" in m[0].content:
            raise LLMError("quota used up")
        return real(task, m, schema)

    llm._fns["story"] = flaky  # type: ignore[attr-defined]
    providers = Providers(llm=llm, tts=FakeTTS(), detector=FixedShots())
    with AppServices(
        cfg.model_copy(update={"media_roots": [movie.parent]}), providers=providers
    ) as services:
        asset = services.library.import_asset(str(movie))
        with Pipeline(services.cfg, providers, db=services.db) as p, pytest.raises(LLMError):
            p.run_stage("analysis.story", asset.id)
        state["fail"] = False
        with Pipeline(services.cfg, providers, db=services.db) as p:
            p.run_stage("analysis.story", asset.id)

        report = services.report.analysis(asset.id)
    db.close()
    story = next(s for s in report.stages if s.stage == "analysis.story")
    assert (story.runs, story.failed_runs, story.cached) == (2, 1, True)
    assert story.llm.calls == 3 and story.llm.failed_calls == 0  # acts twice, then synthesis
    others = [s for s in report.stages if s.stage != "analysis.story"]
    assert all(s.runs == 1 for s in others)


def test_report_for_an_unknown_asset(client: TestClient) -> None:
    r = client.get("/api/assets/ast_missing/report")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_cli_report_prints_a_table(
    cfg: AppConfig,
    recording: Providers,
    movie: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("offscreen.services.pipeline.build_providers", lambda _c, _d: recording)
    monkeypatch.setattr("offscreen.services.jobs.build_providers", lambda _c, _d: recording)
    conf = tmp_path / "config.yaml"
    conf.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())), encoding="utf-8")
    runner = CliRunner()

    ran = runner.invoke(
        app, ["stage", "analysis.story", "--asset", str(movie), "--config", str(conf)]
    )
    assert ran.exit_code == 0, ran.output
    asset = next(line for line in ran.output.splitlines() if line.startswith("asset")).split()[1]

    r = runner.invoke(app, ["report", asset, "--config", str(conf)])
    assert r.exit_code == 0, r.output
    assert f"{asset}  Sample  (complete)" in r.output
    for stage in CHAIN:
        assert stage in r.output
    assert "found  shots 4, transcript lines 6, scenes " in r.output
    assert "model calls (0 failed)" in r.output

    missing = runner.invoke(app, ["report", "ast_missing", "--config", str(conf)])
    assert missing.exit_code == 1 and "error: unknown asset ast_missing" in missing.output
