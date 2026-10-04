from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from offscreen.domain.asset import AudioStream, VideoInfo
from offscreen.domain.common import Rational
from offscreen.domain.plan import (
    AudioRef,
    Clip,
    EditPlan,
    PlanSegment,
    ScriptRef,
    VoiceSpec,
)
from offscreen.domain.timeline import OutputSpec, Timeline
from offscreen.engine import ArtifactStore, Engine, Stage, StageContext, StageOutput
from offscreen.media.probe import ProbeResult
from offscreen.stages.analysis.ingest import ingest
from offscreen.stages.creation.plan import PLAN_FILE
from offscreen.stages.output.compile import TIMELINE_FILE, CompileStage, CompileStageError
from offscreen.store.db import Database
from offscreen.store.files import write_model
from offscreen.store.repos import AssetRepo


class StubPlan(Stage):
    name = "creation.plan"
    version = 1
    lane = "cpu"

    def run(self, ctx: StageContext) -> StageOutput:
        segs = [
            PlanSegment(
                id="seg_01",
                kind="narration",
                text="她走进雪山。小龙倒在地上。",
                voice=VoiceSpec(voice_id="v"),
                audio=AudioRef(file="tts/a.mp3", duration_ms=2000),
                clips=[
                    Clip(
                        asset_id=ctx.scope["asset_id"],
                        shot_id="sh_1",
                        src_in_ms=5000,
                        src_out_ms=7000,
                    )
                ],
            )
        ]
        plan = EditPlan(
            id="pln_t",
            project_id="prj_t",
            version=1,
            author="ai",
            script_ref=ScriptRef(id="scr_t", version=1),
            segments=segs,
        )
        write_model(ctx.out_dir / PLAN_FILE, plan)
        (ctx.out_dir / "tts").mkdir()
        (ctx.out_dir / "tts" / "a.mp3").write_bytes(b"audio-bytes")
        return StageOutput()


def probe_for(width: int, height: int, fps: Rational):  # type: ignore[no-untyped-def]
    def probe(_: Path) -> ProbeResult:
        return ProbeResult(
            duration_ms=90_000,
            video=VideoInfo(width=width, height=height, fps=fps, codec="h264"),
            audio=[AudioStream(index=1, channels=2, sample_rate=48000)],
        )

    return probe


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[AssetRepo]:
    db = Database(tmp_path / "db" / "o.db")
    yield AssetRepo(db)
    db.close()


def register(repo: AssetRepo, tmp_path: Path, **kw: object) -> str:
    movie = tmp_path / "m" / "film.mkv"
    movie.parent.mkdir(exist_ok=True)
    movie.write_bytes(b"x" * 100)
    return ingest(movie, repo, probe_fn=probe_for(**kw)).asset.id  # type: ignore[arg-type]


def test_timeline_keeps_the_source_frame_size_and_rate(repo: AssetRepo, tmp_path: Path) -> None:
    aid = register(repo, tmp_path, width=1920, height=800, fps=Rational(num=24000, den=1001))
    engine = Engine(ArtifactStore(tmp_path / "a"), [StubPlan(), CompileStage(repo)])
    art = engine.ensure("output.compile", {"asset_id": aid})
    tl = art.read_model(TIMELINE_FILE, Timeline)

    assert (tl.output.profile, tl.output.layout) == ("source", "keep")
    assert (tl.output.width, tl.output.height) == (1920, 800)
    assert (tl.output.fps.num, tl.output.fps.den) == (24000, 1001)
    assert tl.duration_frames == 48  # 2 s at 23.976 fps
    assert tl.video[0].src_in_ms == 5000 and [s.text for s in tl.subtitles] == [
        "她走进雪山",
        "小龙倒在地上",
    ]
    assert art.meta["duration_frames"] == 48 and art.meta["subtitles"] == 2
    # The renderer finds the narration next to the timeline.
    assert art.path(tl.narration[0].file).read_bytes() == b"audio-bytes"


@pytest.mark.parametrize(("w", "h", "limit"), [(1920, 800, 22), (1080, 1920, 14), (1000, 1000, 22)])
def test_subtitle_line_limit_follows_orientation(
    repo: AssetRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, w: int, h: int, limit: int
) -> None:
    import offscreen.stages.output.compile as mod

    seen: list[int] = []
    real = mod.compile_timeline

    def spy(plan: EditPlan, output: OutputSpec, *, subtitle_max_chars: int) -> Timeline:
        seen.append(subtitle_max_chars)
        return real(plan, output, subtitle_max_chars=subtitle_max_chars)

    monkeypatch.setattr(mod, "compile_timeline", spy)
    aid = register(repo, tmp_path, width=w, height=h, fps=Rational(num=30, den=1))
    Engine(ArtifactStore(tmp_path / "a"), [StubPlan(), CompileStage(repo)]).ensure(
        "output.compile", {"asset_id": aid}
    )
    assert seen == [limit]


def test_params_carry_the_resolved_output_spec(repo: AssetRepo, tmp_path: Path) -> None:
    a = register(repo, tmp_path, width=1920, height=800, fps=Rational(num=24, den=1))
    p = CompileStage(repo).params({"asset_id": a})["output"]
    assert (p["width"], p["height"], p["fps"], p["layout"]) == (
        1920,
        800,
        {"num": 24, "den": 1},
        "keep",
    )


def test_unknown_asset(repo: AssetRepo) -> None:
    with pytest.raises(CompileStageError, match="unknown asset"):
        CompileStage(repo).params({"asset_id": "ast_nope"})
