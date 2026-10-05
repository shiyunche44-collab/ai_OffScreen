from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from offscreen.domain.index import Scene, Scenes, Shot, Shots
from offscreen.domain.plan import EditPlan
from offscreen.domain.script import Script, ScriptParams, ScriptSegment
from offscreen.engine import ArtifactStore, Engine, Stage, StageContext, StageOutput
from offscreen.providers.adapters.fake import FakeTTS
from offscreen.providers.ports import TTSError
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.creation.plan import AUDIO_DIR, PLAN_FILE, PlanError, PlanStage
from offscreen.stages.creation.script import SCRIPT_FILE
from offscreen.store.files import write_model

ASSET = "ast_t1"
SCOPE = {"asset_id": ASSET}
# 10 shots of 3 s: sh_0000 .. sh_0009. Scene 1 = shots 0-3, scene 2 = shots 4-9.
SHOTS = [Shot(id=f"sh_{i:04d}", start_ms=i * 3000, end_ms=(i + 1) * 3000) for i in range(10)]
SCENES = [
    Scene(id="sc_001", start_ms=0, end_ms=12_000, shot_ids=[s.id for s in SHOTS[:4]], summary="a"),
    Scene(
        id="sc_002", start_ms=12_000, end_ms=30_000, shot_ids=[s.id for s in SHOTS[4:]], summary="b"
    ),
]


def narration(i: int, text: str, refs: list[str]) -> ScriptSegment:
    return ScriptSegment(id=f"seg_{i:02d}", kind="narration", beat="b", text=text, scene_refs=refs)


class StubInputs(Stage):
    """One stage standing in for the three upstream ones (the engine wants three names)."""

    version = 1
    lane = "cpu"

    def __init__(self, name: str, segments: list[ScriptSegment]) -> None:
        self.name = name
        self.segments = segments

    def params(self, scope: Any) -> dict[str, Any]:
        return {"segs": [s.text for s in self.segments]}

    def run(self, ctx: StageContext) -> StageOutput:
        if self.name == "analysis.shots":
            write_model(ctx.out_dir / SHOTS_FILE, Shots(asset_id=ASSET, shots=SHOTS))
        elif self.name == "analysis.scenes":
            write_model(ctx.out_dir / SCENES_FILE, Scenes(asset_id=ASSET, scenes=SCENES))
        else:
            script = Script(
                id="scr_t1",
                project_id="prj_t1",
                version=1,
                author="ai",
                params=ScriptParams(style="s", target_duration_s=10, voice_id="voice-a"),
                segments=self.segments,
            )
            write_model(ctx.out_dir / SCRIPT_FILE, script)
        return StageOutput()


def build(
    tmp_path: Path, segments: list[ScriptSegment], tts: FakeTTS, speed: float = 1.0
) -> Engine:
    return Engine(
        ArtifactStore(tmp_path / "a"),
        [
            StubInputs("creation.script", segments),
            StubInputs("analysis.scenes", segments),
            StubInputs("analysis.shots", segments),
            PlanStage(tts, speed),
        ],
    )


def test_plan_fills_each_narration_with_clips_from_its_scenes(tmp_path: Path) -> None:
    tts = FakeTTS(chars_per_s=4.5)
    segs = [
        narration(1, "字" * 9, ["sc_001"]),
        narration(2, "字" * 27, ["sc_002", "sc_001"]),
    ]
    art = build(tmp_path, segs, tts).ensure("creation.plan", SCOPE)
    plan = art.read_model(PLAN_FILE, EditPlan)

    assert (plan.id, plan.project_id, plan.version, plan.author) == ("pln_t1", "prj_t1", 1, "ai")
    assert (plan.script_ref.id, plan.script_ref.version) == ("scr_t1", 1)
    assert [s.id for s in plan.segments] == ["seg_01", "seg_02"]

    for seg, scenes_ in zip(plan.segments, (SCENES[:1], SCENES[1:] + SCENES[:1]), strict=True):
        assert seg.audio is not None and seg.voice is not None
        assert seg.voice.voice_id == "voice-a"
        # Clips add up to the voice-over, exactly.
        assert sum(c.src_out_ms - c.src_in_ms for c in seg.clips) == seg.audio.duration_ms
        allowed = {sid for sc in scenes_ for sid in sc.shot_ids}
        assert {c.shot_id for c in seg.clips} <= allowed
        assert all(c.asset_id == ASSET and c.speed == 1.0 for c in seg.clips)
        assert seg.source_audio.mode == "duck" and seg.source_audio.gain_db == -20.0
        assert len(seg.audio.char_timings) > 0
        audio_file = art.path(seg.audio.file)
        assert audio_file.parent.name == AUDIO_DIR and audio_file.read_bytes()
    first_audio = plan.segments[0].audio
    assert first_audio is not None and first_audio.duration_ms == 2_000  # 9 chars at 4.5/s
    assert art.meta["segments"] == 2 and art.meta["duration_ms"] == 8_000
    assert tts.calls[0] == ("字" * 9, "voice-a", 1.0)


def test_later_segments_prefer_shots_not_used_before(tmp_path: Path) -> None:
    segs = [narration(1, "字" * 9, ["sc_001"]), narration(2, "字" * 9, ["sc_001"])]
    plan = (
        build(tmp_path, segs, FakeTTS())
        .ensure("creation.plan", SCOPE)
        .read_model(PLAN_FILE, EditPlan)
    )
    first = [c.shot_id for c in plan.segments[0].clips]
    second = [c.shot_id for c in plan.segments[1].clips]
    assert first == ["sh_0000"] and second == ["sh_0001"]


def test_unknown_scene_and_original_segments_fail(tmp_path: Path) -> None:
    with pytest.raises(PlanError, match="sc_404"):
        build(tmp_path, [narration(1, "字" * 9, ["sc_404"])], FakeTTS()).ensure(
            "creation.plan", SCOPE
        )
    orig = ScriptSegment(id="seg_01", kind="original", text="（原声）", line_refs=["ln_0001"])
    with pytest.raises(PlanError, match="original"):
        build(tmp_path, [orig], FakeTTS()).ensure("creation.plan", SCOPE)


def test_tts_failure_propagates_and_leaves_no_artifact(tmp_path: Path) -> None:
    tts = FakeTTS(fail_with=TTSError("boom"))
    with pytest.raises(TTSError):
        build(tmp_path, [narration(1, "字" * 9, ["sc_001"])], tts).ensure("creation.plan", SCOPE)
    assert not list((tmp_path / "a").glob("creation.plan/*/manifest.json"))


def test_cache_key_covers_speed_and_engine(tmp_path: Path) -> None:
    segs = [narration(1, "字" * 9, ["sc_001"])]
    tts = FakeTTS()
    first = build(tmp_path, segs, tts).ensure("creation.plan", SCOPE)
    assert build(tmp_path, segs, tts).ensure("creation.plan", SCOPE).cache_key == first.cache_key
    assert len(tts.calls) == 1
    assert (
        build(tmp_path, segs, tts, speed=1.2).ensure("creation.plan", SCOPE).cache_key
        != first.cache_key
    )
    assert (
        build(tmp_path, segs, FakeTTS(chars_per_s=3.0)).ensure("creation.plan", SCOPE).cache_key
        != first.cache_key
    )


def test_voice_speed_is_validated() -> None:
    with pytest.raises(ValueError):
        PlanStage(FakeTTS(), 3.0)
