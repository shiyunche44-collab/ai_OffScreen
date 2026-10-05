"""The whole chain on a 30 s film with fake LLM / TTS / shot detection and an external
subtitle file: ingest -> proxy -> shots -> transcript -> story -> script -> plan -> compile ->
render. Everything else (ffmpeg, the engine, the stores, the compiler) is real."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from offscreen.cli import app
from offscreen.config import AppConfig
from offscreen.domain.index import Scenes, Story
from offscreen.domain.plan import EditPlan
from offscreen.domain.script import Script
from offscreen.domain.timeline import Timeline
from offscreen.providers.adapters.fake import FakeFaceAnalyzer, FakeTTS
from offscreen.providers.ports import DetectedFace
from offscreen.services.pipeline import Pipeline, Providers, RunOptions

CHAIN = [
    "analysis.proxy", "analysis.shots", "analysis.keyframes", "analysis.transcript",
    "analysis.captions", "analysis.scenes", "analysis.story",
    "creation.script", "creation.plan", "output.compile", "output.render",
]  # fmt: skip
OPTS = RunOptions(minutes=0.25)  # 15 s of commentary


def ffprobe(path: Path) -> dict:  # type: ignore[type-arg]
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
         "-print_format", "json", str(path)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    return json.loads(r.stdout)  # type: ignore[no-any-return]


def artifact_dir(cfg: AppConfig, stage: str) -> Path:
    (d,) = (cfg.data_dir / "artifacts" / stage).iterdir()
    return d


def test_run_all_makes_a_commentary_video_then_reuses_everything(
    cfg: AppConfig, fakes: Providers, movie: Path
) -> None:
    with Pipeline(cfg, fakes) as p:
        first = p.run_all(movie, OPTS)
        assert [r.stage for r in first.stages] == CHAIN
        assert not any(r.cached for r in first.stages)

        # Every layer's document is on disk and validates.
        scenes = Scenes.model_validate_json(
            (artifact_dir(cfg, "analysis.scenes") / "scenes.json").read_bytes()
        )
        story = Story.model_validate_json(
            (artifact_dir(cfg, "analysis.story") / "story.json").read_bytes()
        )
        script = Script.model_validate_json(
            (artifact_dir(cfg, "creation.script") / "script.json").read_bytes()
        )
        plan = EditPlan.model_validate_json(
            (artifact_dir(cfg, "creation.plan") / "plan.json").read_bytes()
        )
        tl = Timeline.model_validate_json(
            (artifact_dir(cfg, "output.compile") / "timeline.json").read_bytes()
        )
        assert scenes.scenes and story.acts and len(script.segments) == 3

        # Chain consistency: scene refs resolve, clips come from the film, voice-over fits.
        scene_ids = {s.id for s in scenes.scenes}
        assert all(set(s.scene_refs) <= scene_ids for s in script.segments)
        assert [s.id for s in plan.segments] == [s.id for s in script.segments]
        assert tl.output.width == 320 and tl.output.height == 180
        assert tl.subtitles, "narration subtitles were compiled"

        # The video is what the timeline says it is.
        info = ffprobe(first.final)
        kinds = {s["codec_type"]: s for s in info["streams"]}
        assert kinds["video"]["nb_read_frames"] == str(tl.duration_frames)
        assert "audio" in kinds
        voice_ms = sum(s.audio.duration_ms for s in plan.segments if s.audio)
        assert abs(float(info["format"]["duration"]) * 1000 - voice_ms) < 150
        assert 12_000 < voice_ms < 18_000  # asked for 15 s

        calls = (len(fakes.llm.calls), len(fakes.tts.calls))  # type: ignore[attr-defined]
        assert calls[0] >= 3 and calls[1] == 3

        # Same command again: every stage cached, no model called, and quick.
        t0 = time.monotonic()
        again = p.run_all(movie, OPTS)
        assert time.monotonic() - t0 < 5
        assert all(r.cached for r in again.stages) and len(again.stages) == len(CHAIN)
        assert again.final == first.final
        assert (len(fakes.llm.calls), len(fakes.tts.calls)) == calls  # type: ignore[attr-defined]

        # A different target length reuses the analysis and redoes the creative stages.
        longer = p.run_all(movie, RunOptions(minutes=0.3))
        cached = {r.stage for r in longer.stages if r.cached}
        assert cached == set(CHAIN[:7])  # all of the analysis


def test_run_stage_by_name_with_a_path_or_an_asset_id(
    cfg: AppConfig, fakes: Providers, movie: Path
) -> None:
    with Pipeline(cfg, fakes) as p:
        by_path = p.run_stage("analysis.shots", str(movie))
        assert [r.stage for r in by_path.stages] == ["analysis.proxy", "analysis.shots"]
        by_id = p.run_stage("analysis.shots", by_path.asset_id)
        assert all(r.cached for r in by_id.stages)
        with pytest.raises(ValueError, match="unknown stage"):
            p.run_stage("analysis.nope", by_path.asset_id)
        with pytest.raises(ValueError, match="unknown asset"):
            p.run_stage("analysis.shots", "ast_missing")


def test_cli_run_all_and_stage(
    cfg: AppConfig, fakes: Providers, movie: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    monkeypatch.setattr("offscreen.services.pipeline.build_providers", lambda _cfg, _db: fakes)
    conf = tmp_path / "config.yaml"
    conf.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())), encoding="utf-8")
    runner = CliRunner()

    r = runner.invoke(app, ["run-all", str(movie), "--minutes", "0.25", "--config", str(conf)])
    assert r.exit_code == 0, r.output
    assert "done    output.render" in r.output and "11 run, 0 cached" in r.output
    final = next(line for line in r.output.splitlines() if line.startswith("video")).split()[-1]
    assert Path(final).is_file() and Path(final).name == "final.mp4"

    again = runner.invoke(app, ["run-all", str(movie), "--minutes", "0.25", "--config", str(conf)])
    assert again.exit_code == 0 and "0 run, 11 cached" in again.output
    asset = next(line for line in again.output.splitlines() if line.startswith("asset")).split()[1]

    s = runner.invoke(app, ["stage", "analysis.story", "--asset", asset, "--config", str(conf)])
    assert s.exit_code == 0 and "cached  analysis.story" in s.output and "video" not in s.output


def test_cli_reports_expected_failures_without_a_traceback(
    cfg: AppConfig, fakes: Providers, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    monkeypatch.setattr("offscreen.services.pipeline.build_providers", lambda _cfg, _db: fakes)
    conf = tmp_path / "config.yaml"
    conf.write_text(yaml.safe_dump(json.loads(cfg.model_dump_json())), encoding="utf-8")
    runner = CliRunner()

    missing = runner.invoke(app, ["run-all", str(tmp_path / "nope.mkv"), "--config", str(conf)])
    assert missing.exit_code == 1 and "error: not a file" in missing.output
    bad_stage = runner.invoke(app, ["stage", "nope", "--asset", "ast_x", "--config", str(conf)])
    assert bad_stage.exit_code == 1 and "unknown stage" in bad_stage.output
    no_conf = runner.invoke(app, ["run-all", "x.mkv", "--config", str(tmp_path / "absent.yaml")])
    assert no_conf.exit_code == 1 and "config file not found" in no_conf.output


def test_tts_is_called_once_per_segment_even_across_runs(
    cfg: AppConfig, fakes: Providers, movie: Path
) -> None:
    assert isinstance(fakes.tts, FakeTTS)
    with Pipeline(cfg, fakes) as p:
        p.run_stage("creation.plan", str(movie), OPTS)
        n = len(fakes.tts.calls)
        p.run_stage("creation.plan", str(movie), OPTS)
    assert n == 3 and len(fakes.tts.calls) == 3


def test_faces_stage_runs_on_the_keyframes_of_the_real_chain(
    cfg: AppConfig, fakes: Providers, movie: Path
) -> None:
    import numpy as np

    from offscreen.domain.index import Faces
    from offscreen.stages.analysis.faces import EMBEDDINGS_FILE, FACES_FILE

    face = DetectedFace(bbox=(0.2, 0.2, 0.5, 0.6), score=0.9, embedding=(1.0, 2.0, 2.0))
    analyzer = FakeFaceAnalyzer(lambda p: [face] if p.name.endswith("_b.jpg") else [])
    fakes.faces = analyzer
    with Pipeline(cfg, fakes) as p:
        result = p.run_stage("analysis.faces", str(movie))
        assert [r.stage for r in result.stages] == [
            "analysis.proxy", "analysis.shots", "analysis.keyframes", "analysis.faces",
        ]  # fmt: skip
        doc = result.artifact.read_model(FACES_FILE, Faces)
        assert [len(s.faces) for s in doc.shots] == [1, 1, 1, 1]  # the middle frame of each shot
        assert all(s.faces[0].frame == 1 for s in doc.shots)
        matrix = np.load(result.artifact.path(EMBEDDINGS_FILE))
        assert matrix.shape == (4, 3)
        np.testing.assert_allclose(matrix[0], [1 / 3, 2 / 3, 2 / 3], rtol=1e-6)
        assert len(analyzer.images) == 12 and all(i.is_file() for i in analyzer.images)

        # the analysis chain itself does not need faces yet
        again = p.run_stage("analysis.story", str(movie))
        assert "analysis.faces" not in [r.stage for r in again.stages]
