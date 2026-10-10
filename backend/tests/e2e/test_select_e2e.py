"""Labelling footage choices and measuring the shot ranking (the 30 s clip, four fixed shots,
captions "雪山里的画面 sh_…" that the scripted model writes)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from offscreen.cli import app
from offscreen.config import AppConfig
from offscreen.domain.index import Captions, Scenes, SelectionLabel, Shots
from offscreen.services.app import AppServices
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.pipeline import Pipeline, Providers
from offscreen.stages.analysis.captions import CAPTIONS_FILE
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE


@pytest.fixture
def services(cfg: AppConfig, fakes: Providers, movie: Path) -> Iterator[AppServices]:
    with AppServices(cfg.model_copy(update={"media_roots": [movie.parent]}), providers=fakes) as s:
        yield s


@pytest.fixture
def asset(services: AppServices, fakes: Providers, movie: Path) -> str:
    a = services.library.import_asset(str(movie))
    with Pipeline(services.cfg, fakes, db=services.db) as p:
        p.run_stage("analysis.scenes", a.id)
    return a.id


def index(services: AppServices, fakes: Providers, asset: str) -> tuple[list[str], list[str]]:
    with Pipeline(services.cfg, fakes, db=services.db) as p:
        shots = p.peek("analysis.shots", asset)
        scenes = p.peek("analysis.scenes", asset)
        caps = p.peek("analysis.captions", asset)
    assert shots and scenes and caps
    assert len(caps.read_model(CAPTIONS_FILE, Captions).captions) == 4
    return (
        [s.id for s in shots.read_model(SHOTS_FILE, Shots).shots],
        [s.id for s in scenes.read_model(SCENES_FILE, Scenes).scenes],
    )


def label(
    lid: str, text: str, scenes: list[str], good: list[str], note: str | None = None
) -> SelectionLabel:
    return SelectionLabel(id=lid, text=text, scene_refs=scenes, acceptable=good, note=note)


def test_labels_are_saved_checked_and_measured(
    services: AppServices, fakes: Providers, asset: str
) -> None:
    shots, scenes = index(services, fakes, asset)
    svc = services.annotations
    assert svc.selection(asset) == []
    with pytest.raises(NotFound, match="no footage choices"):
        svc.evaluate_selection(asset)

    with pytest.raises(InvalidInput, match="unknown shot sh_9999"):
        svc.save_selection(asset, [label("q1", "雪山", scenes, ["sh_9999"])])
    with pytest.raises(InvalidInput, match="unknown scene sc_404"):
        svc.save_selection(asset, [label("q1", "雪山", ["sc_404"], shots[:1])])
    with pytest.raises(InvalidInput, match="duplicate label ids"):
        svc.save_selection(
            asset, [label("q1", "雪山", scenes, shots[:1]), label("q1", "雪", scenes, shots[:1])]
        )

    # q1 talks about the third shot's description; q2 about the first's but accepts only the last
    svc.save_selection(
        asset,
        [
            label("q1", f"雪山里的画面 {shots[2]}", scenes, [shots[2]]),
            label("q2", f"雪山里的画面 {shots[0]}", scenes, [shots[3]], "wrong on purpose"),
        ],
    )
    r = svc.evaluate_selection(asset, k=2)

    assert (r.labels, r.k, r.vector_search) == (2, 2, False)
    by_id = {x.label_id: x for x in r.results}
    assert by_id["q1"].first_rank == 1 and by_id["q1"].first_choice == shots[2]
    assert (by_id["q2"].first_rank or 99) > 1
    assert r.first_choice_rate == 0.5
    assert 0.5 <= r.top_k_hit_rate <= 1.0
    assert svc.selection(asset)[1].note == "wrong on purpose"
    with pytest.raises(InvalidInput, match="k must be"):
        svc.evaluate_selection(asset, k=0)


def test_evaluation_needs_the_analysis(services: AppServices, movie: Path) -> None:
    fresh = services.library.import_asset(str(movie)).id
    with pytest.raises(NotFound, match=r"analysis\.shots has not been built"):
        services.annotations.save_selection(fresh, [])


def test_cli_import_evaluate_and_export(
    cfg: AppConfig,
    fakes: Providers,
    movie: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("offscreen.services.pipeline.build_providers", lambda _c, _d: fakes)
    monkeypatch.setattr("offscreen.services.jobs.build_providers", lambda _c, _d: fakes)
    conf = tmp_path / "config.yaml"
    conf.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())), encoding="utf-8")
    runner = CliRunner()
    ran = runner.invoke(
        app, ["stage", "analysis.scenes", "--asset", str(movie), "--config", str(conf)]
    )
    assert ran.exit_code == 0, ran.output
    asset = next(line for line in ran.output.splitlines() if line.startswith("asset")).split()[1]
    with AppServices(cfg, providers=fakes) as s:
        shots, scenes = index(s, fakes, asset)

    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "id": "q1",
                        "text": f"雪山里的画面 {shots[1]}",
                        "scene_refs": scenes,
                        "acceptable": [shots[1]],
                    }
                ]
            }
        )
    )
    r = runner.invoke(app, ["select", "import", asset, str(labels), "--config", str(conf)])
    assert r.exit_code == 0 and "saved 1 labels" in r.output, r.output

    r = runner.invoke(app, ["select", "evaluate", asset, "-k", "3", "--config", str(conf)])
    assert r.exit_code == 0, r.output
    assert "1 labels, ranked without the vector search" in r.output
    assert "first choice acceptable  1.000" in r.output
    assert "acceptable shot in top 3  1.000" in r.output
    assert "not in the top 3: -" in r.output

    out = runner.invoke(app, ["select", "export", asset, "--config", str(conf)])
    assert json.loads(out.output)["labels"][0]["acceptable"] == [shots[1]]

    labels.write_text('[{"id": "q1"}]')
    bad = runner.invoke(app, ["select", "import", asset, str(labels), "--config", str(conf)])
    assert bad.exit_code == 1 and "error:" in bad.output
    none = runner.invoke(app, ["select", "evaluate", "ast_missing", "--config", str(conf)])
    assert none.exit_code == 1 and "unknown asset" in none.output
