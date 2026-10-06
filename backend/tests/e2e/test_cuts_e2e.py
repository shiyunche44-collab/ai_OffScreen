"""Marking cuts and measuring the detector (the 30 s clip, four fixed shots: cuts at 8 s, 16 s
and 24 s)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from offscreen.api.app import create_app
from offscreen.cli import app
from offscreen.config import AppConfig
from offscreen.services.app import AppServices
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.pipeline import Pipeline, Providers

FPS = 24000 / 1001
TRUE_CUTS = [round(8 * FPS), round(16 * FPS), round(24 * FPS)]  # 192, 384, 576


@pytest.fixture
def services(cfg: AppConfig, fakes: Providers, movie: Path) -> Iterator[AppServices]:
    with AppServices(cfg.model_copy(update={"media_roots": [movie.parent]}), providers=fakes) as s:
        yield s


@pytest.fixture
def client(services: AppServices) -> Iterator[TestClient]:
    with TestClient(create_app(services), raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def asset(services: AppServices, fakes: Providers, movie: Path) -> str:
    a = services.library.import_asset(str(movie))
    with Pipeline(services.cfg, fakes, db=services.db) as p:
        p.run_stage("analysis.shots", a.id)
    return a.id


def test_marks_round_trip_with_the_movies_frame_rate(client: TestClient, asset: str) -> None:
    empty = client.get(f"/api/assets/{asset}/annotations/cuts").json()
    assert (empty["cuts"], empty["marked"], empty["fps_num"], empty["fps_den"]) == (
        [],
        False,
        24000,
        1001,
    )
    r = client.put(f"/api/assets/{asset}/annotations/cuts", json={"cuts": [384, 192, 192, 576]})
    assert r.status_code == 200 and r.json()["cuts"] == [192, 384, 576] and r.json()["marked"]
    assert client.get(f"/api/assets/{asset}/annotations/cuts").json()["cuts"] == [192, 384, 576]
    assert client.put(f"/api/assets/{asset}/annotations/cuts", json={"cuts": []}).json()["marked"]


def test_bad_marks_are_refused(client: TestClient, asset: str) -> None:
    for cuts in ([0, 5], [-3], [10_000]):
        r = client.put(f"/api/assets/{asset}/annotations/cuts", json={"cuts": cuts})
        assert r.status_code == 422, cuts
    assert (
        client.put(f"/api/assets/{asset}/annotations/cuts", json={"cuts": "x"}).status_code == 422
    )
    assert client.get("/api/assets/ast_missing/annotations/cuts").status_code == 404


def test_the_detector_is_scored_against_the_marks(client: TestClient, asset: str) -> None:
    missing = client.get(f"/api/assets/{asset}/annotations/cuts/evaluation")
    assert (
        missing.status_code == 404
        and "no cuts have been marked" in missing.json()["error"]["message"]
    )

    client.put(f"/api/assets/{asset}/annotations/cuts", json={"cuts": TRUE_CUTS})
    ok = client.get(f"/api/assets/{asset}/annotations/cuts/evaluation").json()
    assert (ok["precision"], ok["recall"], ok["f1"]) == (1.0, 1.0, 1.0)
    assert (ok["marked"], ok["detected"], ok["source"], ok["tolerance"]) == (3, 3, "shots", 2)

    # a mark the detector does not have, and one it has but nobody marked
    client.put(f"/api/assets/{asset}/annotations/cuts", json={"cuts": [192, 300, 576]})
    off = client.get(f"/api/assets/{asset}/annotations/cuts/evaluation").json()
    assert (
        off["true_positives"] == 2
        and off["false_positives"] == [384]
        and off["false_negatives"] == [300]
    )
    assert off["f1"] == pytest.approx(0.6667, abs=1e-3)

    # a few frames off: counts only with enough tolerance
    client.put(f"/api/assets/{asset}/annotations/cuts", json={"cuts": [c + 4 for c in TRUE_CUTS]})
    assert client.get(f"/api/assets/{asset}/annotations/cuts/evaluation").json()["f1"] == 0.0
    wide = client.get(f"/api/assets/{asset}/annotations/cuts/evaluation", params={"tolerance": 4})
    assert wide.json()["f1"] == 1.0
    assert (
        client.get(
            f"/api/assets/{asset}/annotations/cuts/evaluation", params={"tolerance": 99}
        ).status_code
        == 422
    )


def test_evaluation_needs_the_shots_and_can_run_the_raw_detector(
    services: AppServices, fakes: Providers, movie: Path
) -> None:
    fresh = services.library.import_asset(str(movie)).id
    services.annotations.save_cuts(fresh, TRUE_CUTS)
    with pytest.raises(NotFound, match=r"analysis\.shots has not been built"):
        services.annotations.evaluate_cuts(fresh)
    with pytest.raises(NotFound, match=r"analysis\.proxy has not been built"):
        services.annotations.evaluate_cuts(fresh, source="detector")
    with pytest.raises(InvalidInput, match="unknown source"):
        services.annotations.evaluate_cuts(fresh, source="magic")

    with Pipeline(services.cfg, fakes, db=services.db) as p:
        p.run_stage("analysis.shots", fresh)
    raw = services.annotations.evaluate_cuts(fresh, source="detector")
    assert raw.source == "detector" and raw.f1 == 1.0


def test_cli_import_evaluate_and_export(
    cfg: AppConfig, fakes: Providers, movie: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("offscreen.services.pipeline.build_providers", lambda _c, _d: fakes)
    monkeypatch.setattr("offscreen.services.jobs.build_providers", lambda _c, _d: fakes)
    conf = tmp_path / "config.yaml"
    conf.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())), encoding="utf-8")
    runner = CliRunner()
    ran = runner.invoke(
        app, ["stage", "analysis.shots", "--asset", str(movie), "--config", str(conf)]
    )
    asset = next(line for line in ran.output.splitlines() if line.startswith("asset")).split()[1]

    marks = tmp_path / "marks.json"
    marks.write_text(json.dumps({"cuts": [*TRUE_CUTS, 100]}))
    r = runner.invoke(app, ["cuts", "import", asset, str(marks), "--config", str(conf)])
    assert r.exit_code == 0 and "saved 4 cuts" in r.output, r.output

    r = runner.invoke(app, ["cuts", "evaluate", asset, "--config", str(conf)])
    assert r.exit_code == 0, r.output
    assert "marked 4   detected 3   matched 3" in r.output
    assert "precision 1.000   recall 0.750   F1 0.857" in r.output
    assert "missed (frames):          100" in r.output
    raw = runner.invoke(
        app, ["cuts", "evaluate", asset, "--raw", "--tolerance", "0", "--config", str(conf)]
    )
    assert raw.exit_code == 0 and "detector  tolerance ±0" in raw.output

    out = runner.invoke(app, ["cuts", "export", asset, "--config", str(conf)])
    assert json.loads(out.output) == {
        "asset_id": asset,
        "fps": [24000, 1001],
        "cuts": sorted([*TRUE_CUTS, 100]),
    }

    marks.write_text('[1, "x"]')
    bad = runner.invoke(app, ["cuts", "import", asset, str(marks), "--config", str(conf)])
    assert bad.exit_code == 1 and "error:" in bad.output
    unknown = runner.invoke(app, ["cuts", "evaluate", "ast_missing", "--config", str(conf)])
    assert unknown.exit_code == 1 and "unknown asset" in unknown.output
