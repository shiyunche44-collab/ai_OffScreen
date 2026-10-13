from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from offscreen.algo.fitting import MAX_CLIP_MS, MAX_SPEED, MIN_SPEED
from offscreen.algo.plan_build import playback_ms
from offscreen.algo.search import search_text
from offscreen.domain.index import (
    Captions,
    Scene,
    Scenes,
    Shot,
    ShotCaption,
    ShotIndex,
    ShotIndexEntry,
    ShotQuality,
    Shots,
    Transcript,
    TranscriptLine,
    Word,
)
from offscreen.domain.plan import Bgm, EditPlan, PlanSegment
from offscreen.domain.script import Script, ScriptParams, ScriptSegment
from offscreen.engine import (
    Artifact,
    ArtifactStore,
    Engine,
    Stage,
    StageCanceled,
    StageContext,
    StageOutput,
)
from offscreen.providers.adapters.fake import FakeEmbedder, FakeTTS
from offscreen.providers.ports import (
    ASR,
    AsrCanceled,
    AsrResult,
    AsrSegment,
    SynthesizedAudio,
    TTSError,
)
from offscreen.stages.analysis.captions import CAPTIONS_FILE
from offscreen.stages.analysis.embeddings import (
    SHOT_INDEX_FILE,
    TEXT_VECTORS_FILE,
)
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.stages.creation.plan import (
    AUDIO_DIR,
    PLAN_FILE,
    PlanError,
    PlanSettings,
    PlanStage,
    PreviousPlan,
)
from offscreen.stages.creation.script import SCRIPT_FILE
from offscreen.store.files import write_model

ASSET = "ast_t1"
SCOPE = {"asset_id": ASSET}
# 10 shots of 3 s: sh_0000 .. sh_0009. Scene 1 = shots 0-3, scene 2 = shots 4-9.
SHOTS = [
    Shot(
        id=f"sh_{i:04d}",
        start_ms=i * 3000,
        end_ms=(i + 1) * 3000,
        quality=ShotQuality(sharpness=0.7, brightness=0.6),
    )
    for i in range(10)
]
SCENES = [
    Scene(id="sc_001", start_ms=0, end_ms=12_000, shot_ids=[s.id for s in SHOTS[:4]], summary="a"),
    Scene(
        id="sc_002", start_ms=12_000, end_ms=30_000, shot_ids=[s.id for s in SHOTS[4:]], summary="b"
    ),
]
CAPTION_OF = {s.id: f"画面{i}号镜头里的内容各不相同{i}{i}" for i, s in enumerate(SHOTS)}
CAPTIONS = [ShotCaption(shot_id=sid, caption=text) for sid, text in CAPTION_OF.items()]
LINES = [
    TranscriptLine(id="ln_0001", start_ms=20_000, end_ms=22_000, text="你是谁"),
    TranscriptLine(id="ln_0002", start_ms=25_000, end_ms=26_000, text="我是龙"),
]


def narration(i: int, text: str, refs: list[str]) -> ScriptSegment:
    return ScriptSegment(id=f"seg_{i:02d}", kind="narration", beat="b", text=text, scene_refs=refs)


def script_of(segments: list[ScriptSegment], voice: str = "voice-a", version: int = 1) -> Script:
    return Script(
        id="scr_t1",
        project_id="prj_t1",
        version=version,
        author="ai",
        params=ScriptParams(style="s", target_duration_s=10, voice_id=voice),
        segments=segments,
    )


class StubInputs(Stage):
    """One stage standing in for each upstream one."""

    version = 1
    lane = "cpu"

    def __init__(self, name: str, script: Script, vectors: bool = False) -> None:
        self.name = name
        self.script = script
        self.vectors = vectors

    def params(self, scope: Any) -> dict[str, Any]:
        return {"segs": [s.text for s in self.script.segments]}

    def run(self, ctx: StageContext) -> StageOutput:
        out = ctx.out_dir
        match self.name:
            case "analysis.shots":
                write_model(out / SHOTS_FILE, Shots(asset_id=ASSET, shots=SHOTS))
            case "analysis.scenes":
                write_model(out / SCENES_FILE, Scenes(asset_id=ASSET, scenes=SCENES))
            case "analysis.captions":
                write_model(out / CAPTIONS_FILE, Captions(asset_id=ASSET, captions=CAPTIONS))
            case "analysis.transcript":
                write_model(
                    out / TRANSCRIPT_FILE,
                    Transcript(asset_id=ASSET, language="zh", source="test", lines=LINES),
                )
            case "analysis.embeddings":
                embedder = FakeEmbedder()
                entries = [
                    ShotIndexEntry(
                        shot_id=s.id,
                        start_ms=s.start_ms,
                        end_ms=s.end_ms,
                        caption=search_text(c),
                    )
                    for s, c in zip(SHOTS, CAPTIONS, strict=True)
                ]
                write_model(
                    out / SHOT_INDEX_FILE,
                    ShotIndex(asset_id=ASSET, text_model="fake", shots=entries),
                )
                rows = embedder.embed_texts([e.caption for e in entries])
                np.save(out / TEXT_VECTORS_FILE, np.asarray(rows, dtype=np.float32))
            case _:
                write_model(out / SCRIPT_FILE, self.script)
        return StageOutput()


def build(
    tmp_path: Path,
    segments: list[ScriptSegment],
    tts: FakeTTS,
    speed: float = 1.0,
    settings: PlanSettings | None = None,
    embed: bool = False,
    voice: str = "voice-a",
    aligner: ASR | None = None,
) -> Engine:
    script = settings.script if settings and settings.script else script_of(segments, voice)
    names = ["creation.script", "analysis.scenes", "analysis.shots", "analysis.captions"]
    names += ["analysis.transcript"] + (["analysis.embeddings"] if embed else [])
    return Engine(
        ArtifactStore(tmp_path / "a"),
        [
            *(StubInputs(n, script) for n in names),
            PlanStage(
                tts,
                speed,
                settings,
                text_embedder=FakeEmbedder() if embed else None,
                aligner=aligner,
            ),
        ],
    )


def plan_of(art: Artifact) -> EditPlan:
    return art.read_model(PLAN_FILE, EditPlan)


def total(seg: PlanSegment) -> int:
    return sum(playback_ms(c) for c in seg.clips)


def rebuild_on(
    tmp_path: Path,
    prev: Artifact,
    segments: list[ScriptSegment],
    tts: FakeTTS,
    *,
    version: int = 2,
    voice: str = "voice-a",
    plan: EditPlan | None = None,
) -> Artifact:
    """Build again on top of the plan of `prev`, following a script with `segments`."""
    settings = PlanSettings(
        script=script_of(segments, voice, version),
        previous=PreviousPlan(plan or plan_of(prev), prev.dir),
    )
    return build(tmp_path, segments, tts, settings=settings).ensure("creation.plan", SCOPE)


def test_plan_fills_each_voice_over_exactly_with_footage_from_its_scenes(tmp_path: Path) -> None:
    tts = FakeTTS(chars_per_s=4.5)
    segs = [
        narration(1, "字" * 9, ["sc_001"]),
        narration(2, "字" * 54, ["sc_002", "sc_001"]),  # 12 s: more than one clip's worth
    ]
    art = build(tmp_path, segs, tts).ensure("creation.plan", SCOPE)
    plan = plan_of(art)

    assert (plan.id, plan.project_id, plan.version, plan.author) == ("pln_t1", "prj_t1", 1, "ai")
    assert plan.parent_version is None
    assert (plan.script_ref.id, plan.script_ref.version) == ("scr_t1", 1)
    assert [s.id for s in plan.segments] == ["seg_01", "seg_02"]

    for seg in plan.segments:
        assert seg.audio is not None and seg.voice is not None
        assert seg.voice.voice_id == "voice-a" and seg.stale is False
        assert total(seg) == seg.audio.duration_ms  # to the millisecond
        for c in seg.clips:
            assert MIN_SPEED - 1e-9 <= c.speed <= MAX_SPEED + 1e-9
            assert c.src_out_ms - c.src_in_ms <= MAX_CLIP_MS
            assert c.asset_id == ASSET and c.score is not None and 0 <= c.score <= 1
            assert c.locked is False
        assert seg.source_audio.mode == "duck" and seg.source_audio.gain_db == -20.0
        assert len(seg.audio.char_timings) > 0
        audio_file = art.path(seg.audio.file)
        assert audio_file.parent.name == AUDIO_DIR and audio_file.read_bytes()
    first, second = plan.segments
    assert first.audio is not None and first.audio.duration_ms == 2_000  # 9 chars at 4.5/s
    assert {c.shot_id for c in first.clips} <= {s.id for s in SHOTS[:4]}
    assert len(second.clips) >= 3
    assert art.meta["segments"] == 2 and art.meta["duration_ms"] == 14_000
    assert tts.calls[0] == ("字" * 9, "voice-a", 1.0)


def test_footage_runs_forward_in_film_time(tmp_path: Path) -> None:
    plan = plan_of(
        build(tmp_path, [narration(1, "字" * 54, ["sc_002"])], FakeTTS()).ensure(
            "creation.plan", SCOPE
        )
    )

    starts = [c.src_in_ms for c in plan.segments[0].clips]
    assert starts == sorted(starts)


def test_later_segments_prefer_shots_not_used_before(tmp_path: Path) -> None:
    segs = [narration(1, "字" * 9, ["sc_001"]), narration(2, "字" * 9, ["sc_001"])]
    plan = plan_of(build(tmp_path, segs, FakeTTS()).ensure("creation.plan", SCOPE))

    first = {c.shot_id for c in plan.segments[0].clips}
    second = {c.shot_id for c in plan.segments[1].clips}
    assert first and second and not first & second


def test_a_caption_that_matches_the_text_wins(tmp_path: Path) -> None:
    # sh_0002's description is the only one the text talks about.
    segs = [narration(1, CAPTION_OF["sh_0002"], ["sc_001"])]
    plan = plan_of(build(tmp_path, segs, FakeTTS()).ensure("creation.plan", SCOPE))

    assert "sh_0002" in {c.shot_id for c in plan.segments[0].clips}


def test_credits_shots_are_never_chosen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captions = [
        c.model_copy(update={"is_credits": c.shot_id != "sh_0003"}) for c in CAPTIONS
    ]  # in scene 1 only sh_0003 is not credits
    monkeypatch.setattr(sys.modules[__name__], "CAPTIONS", captions)

    plan = plan_of(
        build(tmp_path, [narration(1, "字" * 9, ["sc_001"])], FakeTTS()).ensure(
            "creation.plan", SCOPE
        )
    )

    assert {c.shot_id for c in plan.segments[0].clips} == {"sh_0003"}


def test_original_sound_segment_plays_the_cited_lines_and_narration_keeps_off_it(
    tmp_path: Path,
) -> None:
    orig = ScriptSegment(id="seg_02", kind="original", text="（原声）", line_refs=["ln_0001"])
    # Scene 2 holds the lines (20.0 s - 22.0 s, padded to 19.8 s - 22.2 s = shots 6 and 7).
    segs = [narration(1, "字" * 54, ["sc_002"]), orig]
    art = build(tmp_path, segs, FakeTTS()).ensure("creation.plan", SCOPE)
    plan = plan_of(art)

    original = plan.segments[1]
    assert original.kind == "original" and original.audio is None
    (c,) = original.clips
    assert (c.src_in_ms, c.src_out_ms, c.locked) == (19_800, 22_200, True)
    assert original.source_audio.mode == "full"
    for clip in plan.segments[0].clips:
        assert clip.src_out_ms <= 19_800 or clip.src_in_ms >= 22_200
    assert art.meta["warnings"] == []


def test_unknown_scene_and_unknown_lines_fail(tmp_path: Path) -> None:
    with pytest.raises(PlanError, match="sc_404"):
        build(tmp_path, [narration(1, "字" * 9, ["sc_404"])], FakeTTS()).ensure(
            "creation.plan", SCOPE
        )
    orig = ScriptSegment(id="seg_01", kind="original", text="（原声）", line_refs=["ln_404"])
    with pytest.raises(PlanError, match="ln_404"):
        build(tmp_path, [orig], FakeTTS()).ensure("creation.plan", SCOPE)


def test_the_engine_hears_numbers_spelled_out_and_subtitles_keep_the_script_text(
    tmp_path: Path,
) -> None:
    tts = FakeTTS()
    text = "他在1990年出生，花了$2亿。"
    plan = plan_of(
        build(tmp_path, [narration(1, text, ["sc_001"])], tts).ensure("creation.plan", SCOPE)
    )
    assert tts.calls[0][0] == "他在一九九零年出生，花了两亿美元。"
    (seg,) = plan.segments
    assert seg.text == text and seg.audio is not None
    timings = seg.audio.char_timings
    assert len(timings) == len(text)
    assert all(a <= b <= c for (a, b), (c, _) in itertools.pairwise(timings))
    assert timings[-1][1] <= seg.audio.duration_ms
    year = timings[2:6]  # "1990" shares the time of 一九九零
    assert year[0][0] < year[-1][1] and year[-1][1] - year[0][0] >= 600


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


# ---- incremental build ------------------------------------------------------------------
SEGS = [
    narration(1, "字" * 18, ["sc_001"]),
    narration(2, "字" * 27, ["sc_002"]),
    narration(3, "甲" * 18, ["sc_001", "sc_002"]),
]


def test_an_edited_segment_is_rebuilt_and_the_others_are_carried_over(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    before = plan_of(first)

    tts = FakeTTS()
    edited = [SEGS[0], narration(2, "乙" * 36, ["sc_002"]), SEGS[2]]
    art = rebuild_on(tmp_path, first, edited, tts)
    plan = plan_of(art)

    assert [t for t, _, _ in tts.calls] == ["乙" * 36]  # only the edited text was spoken
    assert art.meta["rebuilt"] == {"seg_02": "text"}
    assert art.meta["reused"] == ["seg_01", "seg_03"] and art.meta["removed"] == []
    assert plan.segments[0] == before.segments[0] and plan.segments[2] == before.segments[2]
    changed = plan.segments[1]
    assert changed.text == "乙" * 36 and changed.audio is not None
    assert changed.audio != before.segments[1].audio
    assert total(changed) == changed.audio.duration_ms
    # Every audio file the new plan refers to is in the new artifact.
    for seg in plan.segments:
        assert seg.audio is not None and art.path(seg.audio.file).read_bytes()
    assert (plan.version, plan.parent_version, plan.id) == (2, 1, before.id)
    assert plan.script_ref.version == 2


def test_the_rebuild_log_names_the_segments_and_the_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)

    with caplog.at_level(logging.INFO, logger="offscreen.stages.creation.plan"):
        rebuild_on(
            tmp_path, first, [SEGS[0], narration(2, "乙" * 36, ["sc_002"]), SEGS[2]], FakeTTS()
        )

    lines = [r.getMessage() for r in caplog.records]
    assert "plan: 2 reused, 1 rebuilt" in lines
    assert "plan: rebuilt seg_02 (text)" in lines and "plan: reused seg_01" in lines


def test_nothing_changed_means_nothing_is_spoken_again(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)

    tts = FakeTTS()
    art = rebuild_on(tmp_path, first, SEGS, tts, version=1)

    assert tts.calls == []
    assert art.meta["rebuilt"] == {} and len(art.meta["reused"]) == 3
    # The very same plan, so everything built from it stays cached.
    assert plan_of(art) == plan_of(first)
    assert art.content_hash == first.content_hash


def test_new_removed_and_reordered_segments(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)

    tts = FakeTTS()
    new = narration(4, "丙" * 18, ["sc_002"])
    art = rebuild_on(tmp_path, first, [SEGS[2], new, SEGS[0]], tts)  # seg_02 dropped

    plan = plan_of(art)
    assert [s.id for s in plan.segments] == ["seg_03", "seg_04", "seg_01"]
    assert art.meta["rebuilt"] == {"seg_04": "new"}
    assert art.meta["reused"] == ["seg_03", "seg_01"] and art.meta["removed"] == ["seg_02"]
    assert [t for t, _, _ in tts.calls] == ["丙" * 18]


def test_a_new_voice_rebuilds_every_segment(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)

    tts = FakeTTS()
    art = rebuild_on(tmp_path, first, SEGS, tts, voice="voice-b")

    assert art.meta["rebuilt"] == {s.id: "voice" for s in SEGS}
    assert {v for _, v, _ in tts.calls} == {"voice-b"}


def test_a_stale_segment_is_rebuilt(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    before = plan_of(first)
    stale = before.model_copy(
        update={
            "segments": [
                before.segments[0],
                before.segments[1].model_copy(update={"stale": True}),
                before.segments[2],
            ]
        }
    )

    art = rebuild_on(tmp_path, first, SEGS, FakeTTS(), plan=stale)

    assert art.meta["rebuilt"] == {"seg_02": "stale"}
    assert plan_of(art).segments[1].stale is False


def test_a_missing_audio_file_is_spoken_again(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    audio = plan_of(first).segments[0].audio
    assert audio is not None
    settings = PlanSettings(
        script=script_of(SEGS, version=2),
        previous=PreviousPlan(plan_of(first), tmp_path / "gone"),  # no audio there
    )

    art = build(tmp_path, SEGS, FakeTTS(), settings=settings).ensure("creation.plan", SCOPE)

    assert art.meta["rebuilt"] == {s.id: "audio_missing" for s in SEGS}


def test_locked_footage_survives_a_rebuild(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    before = plan_of(first)
    pinned = before.segments[1].clips[0].model_copy(update={"locked": True})
    locked_plan = before.model_copy(
        update={
            "segments": [
                before.segments[0],
                before.segments[1].model_copy(update={"clips": [pinned]}),
                before.segments[2],
            ]
        }
    )

    edited = [SEGS[0], narration(2, "乙" * 54, ["sc_002"]), SEGS[2]]  # 12 s of voice-over
    art = rebuild_on(tmp_path, first, edited, FakeTTS(), plan=locked_plan)

    seg = plan_of(art).segments[1]
    assert pinned in seg.clips
    assert seg.audio is not None
    assert total(seg) == seg.audio.duration_ms  # the new footage fills what the lock leaves
    assert len(seg.clips) > 1


def test_locked_footage_longer_than_the_voice_over_needs_no_new_footage(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    before = plan_of(first)
    long_lock = (
        before.segments[0]
        .clips[0]
        .model_copy(update={"locked": True, "src_in_ms": 0, "src_out_ms": 3000, "speed": 1.0})
    )
    locked_plan = before.model_copy(
        update={
            "segments": [
                before.segments[0].model_copy(update={"clips": [long_lock]}),
                *before.segments[1:],
            ]
        }
    )

    edited = [narration(1, "字" * 9, ["sc_001"]), *SEGS[1:]]  # a 2 s voice-over
    art = rebuild_on(tmp_path, first, edited, FakeTTS(), plan=locked_plan)

    assert plan_of(art).segments[0].clips == [long_lock]


def test_locked_footage_is_not_picked_again_by_other_segments(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    before = plan_of(first)
    lock = before.segments[2].clips[0].model_copy(update={"locked": True})
    locked_plan = before.model_copy(
        update={
            "segments": [
                *before.segments[:2],
                before.segments[2].model_copy(update={"clips": [lock]}),
            ]
        }
    )

    edited = [narration(1, "丁" * 18, ["sc_001", "sc_002"]), *SEGS[1:]]
    art = rebuild_on(tmp_path, first, edited, FakeTTS(), plan=locked_plan)

    assert lock.shot_id not in {c.shot_id for c in plan_of(art).segments[0].clips}


def test_a_rebuild_does_not_touch_bgm_or_output_profile(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    custom = plan_of(first).model_copy(
        update={"bgm": Bgm(file="bgm/x.mp3"), "output_profile": "vertical_1080"}
    )

    plan = plan_of(rebuild_on(tmp_path, first, SEGS, FakeTTS(), plan=custom))

    assert plan.bgm == custom.bgm and plan.output_profile == "vertical_1080"


def test_the_edited_script_replaces_the_generated_one(tmp_path: Path) -> None:
    generated = [narration(1, "字" * 9, ["sc_001"])]
    edited = script_of([narration(1, "人工改过的文案" * 2, ["sc_001"])], version=5)
    settings = PlanSettings(script=edited)

    art = build(tmp_path, generated, FakeTTS(), settings=settings).ensure("creation.plan", SCOPE)

    plan = plan_of(art)
    assert plan.segments[0].text == "人工改过的文案" * 2
    assert plan.script_ref.version == 5
    # The generated script is not an input at all then.
    assert "creation.script" not in {
        r.stage for r in PlanStage(FakeTTS(), 1.0, settings).inputs({})
    }


def test_cache_key_covers_the_script_and_the_plan_built_on(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    keys = {first.cache_key}
    again = rebuild_on(tmp_path, first, SEGS, FakeTTS())
    keys.add(again.cache_key)
    keys.add(rebuild_on(tmp_path, first, SEGS, FakeTTS(), version=3).cache_key)

    assert len(keys) == 3
    assert rebuild_on(tmp_path, first, SEGS, FakeTTS()).cache_key == again.cache_key


def test_the_vector_search_brings_in_footage_beyond_the_cited_scenes(tmp_path: Path) -> None:
    # The text is the description of sh_0007 (scene 2), but the segment cites scene 1 only.
    segs = [narration(1, search_text(CAPTIONS[7]), ["sc_001"])]

    without = plan_of(build(tmp_path, segs, FakeTTS()).ensure("creation.plan", SCOPE))
    with_search = plan_of(
        build(tmp_path, segs, FakeTTS(), embed=True).ensure("creation.plan", SCOPE)
    )

    assert "sh_0007" not in {c.shot_id for c in without.segments[0].clips}
    assert "sh_0007" in {c.shot_id for c in with_search.segments[0].clips}


# ---- what a person's edits of the plan mean to a rebuild ----------------------------------
def with_segments(prev: EditPlan, segments: list[PlanSegment]) -> EditPlan:
    return prev.model_copy(update={"segments": segments})


def test_a_voice_a_person_chose_is_spoken_by_the_next_build_and_kept_after_that(
    tmp_path: Path,
) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    before = plan_of(first)
    mine = before.segments[1].voice.model_copy(update={"voice_id": "mine", "speed": 1.25})  # type: ignore[union-attr]
    edited = with_segments(
        before,
        [
            before.segments[0],
            before.segments[1].model_copy(
                update={"voice": mine, "voice_pinned": True, "stale": True}
            ),
            before.segments[2],
        ],
    )

    tts = FakeTTS()
    art = rebuild_on(tmp_path, first, SEGS, tts, plan=edited)

    assert tts.calls == [("字" * 27, "mine", 1.25)]
    seg = plan_of(art).segments[1]
    assert (seg.voice, seg.voice_pinned, seg.stale) == (mine, True, False)
    assert art.meta["rebuilt"] == {"seg_02": "stale"}
    # a later build does not take it back to the script's voice
    tts2 = FakeTTS()
    again = rebuild_on(tmp_path, art, SEGS, tts2, version=2)
    assert tts2.calls == [] and plan_of(again).segments[1].voice == mine


def test_a_trimmed_original_sound_segment_is_kept_until_the_script_cites_other_lines(
    tmp_path: Path,
) -> None:
    orig = ScriptSegment(id="seg_02", kind="original", text="（原声）", line_refs=["ln_0001"])
    segs = [narration(1, "字" * 9, ["sc_001"]), orig]
    first = build(tmp_path, segs, FakeTTS()).ensure("creation.plan", SCOPE)
    before = plan_of(first)
    assert before.segments[1].line_refs == ["ln_0001"]
    trimmed = (
        before.segments[1].clips[0].model_copy(update={"src_in_ms": 20_500, "src_out_ms": 21_500})
    )
    edited = with_segments(
        before, [before.segments[0], before.segments[1].model_copy(update={"clips": [trimmed]})]
    )

    kept = rebuild_on(tmp_path, first, segs, FakeTTS(), plan=edited)
    assert plan_of(kept).segments[1].clips == [trimmed]

    other = ScriptSegment(id="seg_02", kind="original", text="（原声）", line_refs=["ln_0002"])
    redone = rebuild_on(tmp_path, first, [segs[0], other], FakeTTS(), plan=edited)
    (c,) = plan_of(redone).segments[1].clips
    assert (c.src_in_ms, c.src_out_ms) == (24_800, 26_200)


def test_the_order_and_deletions_a_person_made_in_the_plan_survive_a_rebuild(
    tmp_path: Path,
) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    before = plan_of(first)
    s1, _s2, s3 = before.segments
    edited = with_segments(before, [s3, s1])  # moved seg_03 to the front, deleted seg_02
    new = narration(4, "丙" * 18, ["sc_002"])
    script = [*SEGS, new]  # the script gains seg_04 at the end
    settings = PlanSettings(
        script=script_of(script, version=2),
        previous=PreviousPlan(edited, first.dir),
        previous_script_ids=tuple(s.id for s in SEGS),
    )

    tts = FakeTTS()
    art = build(tmp_path, script, tts, settings=settings).ensure("creation.plan", SCOPE)

    plan = plan_of(art)
    # seg_04 follows seg_03 in the script, so it goes after seg_03 wherever the person put it
    assert [s.id for s in plan.segments] == ["seg_03", "seg_04", "seg_01"]
    assert art.meta["rebuilt"] == {"seg_04": "new"} and art.meta["removed"] == []
    assert [t for t, _, _ in tts.calls] == ["丙" * 18]
    assert plan.segments[0] == s3 and plan.segments[2] == s1


def test_original_sound_inserted_by_a_person_stays(tmp_path: Path) -> None:
    first = build(tmp_path, SEGS, FakeTTS()).ensure("creation.plan", SCOPE)
    before = plan_of(first)
    inserted = PlanSegment(
        id="seg_o01",
        kind="original",
        line_refs=["ln_0001"],
        clips=[before.segments[0].clips[0].model_copy(update={"locked": True})],
    )
    edited = with_segments(before, [before.segments[0], inserted, *before.segments[1:]])
    settings = PlanSettings(
        script=script_of(SEGS, version=1),
        previous=PreviousPlan(edited, first.dir),
        previous_script_ids=tuple(s.id for s in SEGS),
    )

    art = build(tmp_path, SEGS, FakeTTS(), settings=settings).ensure("creation.plan", SCOPE)

    assert [s.id for s in plan_of(art).segments] == ["seg_01", "seg_o01", "seg_02", "seg_03"]
    assert plan_of(art).segments[1] == inserted
    assert art.meta["rebuilt"] == {}


# ---- timings by ASR for engines that give none (M6-05) -----------------------------------------
class SilentTimingsTTS(FakeTTS):
    """An engine that gives no timestamps (as CosyVoice)."""

    def synthesize(self, text: str, *, voice_id: str, speed: float = 1.0) -> SynthesizedAudio:
        audio = super().synthesize(text, voice_id=voice_id, speed=speed)
        return SynthesizedAudio(
            audio.data, audio.format, audio.sample_rate, audio.duration_ms, char_timings=[]
        )


class ScriptedAsr:
    """Hears `heard` (word, start, end) in whatever it is given; records the wavs."""

    id = "scripted-asr"

    def __init__(self, heard: list[tuple[str, int, int]], fail: Exception | None = None) -> None:
        self.heard = heard
        self.fail = fail
        self.calls: list[tuple[int, str | None]] = []

    def transcribe(
        self,
        wav: Path,
        *,
        language: str | None = None,
        on_progress: Any = None,
        should_cancel: Any = None,
    ) -> AsrResult:
        import wave

        with wave.open(str(wav)) as w:
            assert (w.getframerate(), w.getnchannels()) == (16000, 1)
        self.calls.append((wav.stat().st_size, language))
        if self.fail is not None:
            raise self.fail
        words = [Word(w=w, start_ms=s, end_ms=e) for w, s, e in self.heard]
        text = "".join(w for w, _, _ in self.heard)
        return AsrResult("zh", [AsrSegment(0, self.heard[-1][2], text, words)])


ALIGN_TEXT = "他在1990年出生，后来回到小镇。"  # spoken: 他在一九九零年出生，后来回到小镇。


def test_voice_over_without_timestamps_is_transcribed_for_subtitles(tmp_path: Path) -> None:
    heard = [
        ("他在", 0, 400),
        ("1990年", 400, 1400),
        ("出声", 1400, 1800),
        ("后来回到小镇", 1800, 3000),
    ]
    asr = ScriptedAsr(heard)
    art = build(
        tmp_path, [narration(1, ALIGN_TEXT, ["sc_001"])], SilentTimingsTTS(), aligner=asr
    ).ensure("creation.plan", SCOPE)
    (seg,) = plan_of(art).segments
    assert seg.audio is not None and asr.calls and asr.calls[0][1] == "zh"
    t = seg.audio.char_timings
    assert len(t) == len(ALIGN_TEXT)
    assert t[0] == (0, 200) and t[1] == (200, 400)
    assert t[2][0] == 400 and t[5][1] == 1200  # "1990" was heard as 一九九零
    assert t[-2] == (2800, 3000)
    assert art.meta["warnings"] == []
    assert not list(art.dir.glob("**/speech.wav"))  # the 16 kHz copy is not kept


def test_engine_timestamps_are_used_as_they_are(tmp_path: Path) -> None:
    asr = ScriptedAsr([("字", 0, 100)])
    build(tmp_path, [narration(1, ALIGN_TEXT, ["sc_001"])], FakeTTS(), aligner=asr).ensure(
        "creation.plan", SCOPE
    )
    assert asr.calls == []


@pytest.mark.parametrize(
    ("asr", "why"),
    [
        (ScriptedAsr([("完全不同的话", 0, 2000)]), "recognized"),
        (
            ScriptedAsr([("x", 0, 1)], fail=RuntimeError("faster-whisper is not installed")),
            "failed",
        ),
    ],
)
def test_a_poor_or_failed_transcription_leaves_no_timings_and_says_so(
    tmp_path: Path, asr: ScriptedAsr, why: str
) -> None:
    art = build(
        tmp_path, [narration(1, ALIGN_TEXT, ["sc_001"])], SilentTimingsTTS(), aligner=asr
    ).ensure("creation.plan", SCOPE)
    (seg,) = plan_of(art).segments
    assert seg.audio is not None and seg.audio.char_timings == []
    (warning,) = art.meta["warnings"]
    assert warning.startswith("seg_01: no subtitle timings") and why in warning


def test_canceling_during_transcription_cancels_the_stage(tmp_path: Path) -> None:
    asr = ScriptedAsr([("x", 0, 1)], fail=AsrCanceled("stop"))
    with pytest.raises(StageCanceled):
        build(
            tmp_path, [narration(1, ALIGN_TEXT, ["sc_001"])], SilentTimingsTTS(), aligner=asr
        ).ensure("creation.plan", SCOPE)


def test_the_aligner_is_part_of_the_cache_key(tmp_path: Path) -> None:
    segs = [narration(1, ALIGN_TEXT, ["sc_001"])]
    tts = SilentTimingsTTS()
    plain = build(tmp_path, segs, tts).ensure("creation.plan", SCOPE)
    aligned = build(tmp_path, segs, tts, aligner=ScriptedAsr([("他", 0, 100)])).ensure(
        "creation.plan", SCOPE
    )
    assert plain.cache_key != aligned.cache_key
