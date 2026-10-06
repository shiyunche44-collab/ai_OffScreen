from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from offscreen import styles
from offscreen.algo.script_rules import KnownName
from offscreen.domain.index import Act, Scene, Scenes, Story, Transcript, TranscriptLine
from offscreen.domain.script import OutlineBeat, Script, ScriptOutline
from offscreen.engine import ArtifactStore, Engine, Stage, StageContext, StageOutput
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.providers.ports import LLMError
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.story import STORY_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.stages.creation.outline import OUTLINE_FILE
from offscreen.stages.creation.script import (
    SCRIPT_FILE,
    ScriptError,
    ScriptSettings,
    ScriptStage,
)
from offscreen.store.files import write_model

ASSET = "ast_t1"
SCOPE = {"asset_id": ASSET}
MODELS = {"script_write": "fake/m"}
STYLE = "suspense"


class StubStory(Stage):
    name = "analysis.story"
    version = 1
    lane = "api"

    def run(self, ctx: StageContext) -> StageOutput:
        story = Story(
            asset_id=ASSET,
            logline="女孩寻龙",
            synopsis="一个女孩为了找回小龙走遍世界。",
            acts=[Act(name="一", scene_ids=["sc_001", "sc_002"], summary="x")],
            ending="小龙已死",
            themes=["失去"],
        )
        write_model(ctx.out_dir / STORY_FILE, story)
        return StageOutput()


class StubScenes(Stage):
    name = "analysis.scenes"
    version = 1
    lane = "api"

    def run(self, ctx: StageContext) -> StageOutput:
        scenes = Scenes(
            asset_id=ASSET,
            scenes=[
                Scene(id="sc_001", start_ms=0, end_ms=30_000, shot_ids=["sh_1"], summary="相遇"),
                Scene(
                    id="sc_002", start_ms=30_000, end_ms=60_000, shot_ids=["sh_2"], summary="分别"
                ),
            ],
        )
        write_model(ctx.out_dir / SCENES_FILE, scenes)
        return StageOutput()


class StubTranscript(Stage):
    name = "analysis.transcript"
    version = 1
    lane = "cpu"

    def run(self, ctx: StageContext) -> StageOutput:
        lines = [
            TranscriptLine(id="ln_1", start_ms=1_000, end_ms=3_000, text="我们不能留在这里。"),
            TranscriptLine(id="ln_2", start_ms=40_000, end_ms=42_000, text="再见了，小龙。"),
        ]
        doc = Transcript(asset_id=ASSET, language="zh", source="subtitle:external", lines=lines)
        write_model(ctx.out_dir / TRANSCRIPT_FILE, doc)
        return StageOutput()


class StubOutline(Stage):
    name = "creation.outline"
    version = 1
    lane = "api"

    def run(self, ctx: StageContext) -> StageOutput:
        write_model(ctx.out_dir / OUTLINE_FILE, OUTLINE)
        return StageOutput()


OUTLINE = ScriptOutline(
    asset_id=ASSET,
    style=STYLE,
    target_duration_s=20,
    beats=[
        OutlineBeat(beat="hook", scene_refs=["sc_001"], target_s=10, focus="开场的相遇"),
        OutlineBeat(beat="ending", scene_refs=["sc_002"], target_s=10, focus="分别与结局"),
    ],
)


def seg(chars: int, refs: list[str]) -> dict[str, Any]:
    return {"text": "字" * chars + "。", "scene_refs": refs}


# 10 s * 4.5 = 45 chars per beat, allowed 36-54.
HOOK = {"segments": [seg(45, ["sc_001"])]}
ENDING = {"segments": [seg(20, ["sc_002"]), seg(25, ["sc_002", "sc_001"])]}
SETTINGS = ScriptSettings(target_duration_s=20, voice_id="v1", style=STYLE)


def build(tmp_path: Path, llm: FakeLLM, settings: ScriptSettings = SETTINGS) -> Engine:
    return Engine(
        ArtifactStore(tmp_path / "a"),
        [
            StubStory(),
            StubScenes(),
            StubTranscript(),
            StubOutline(),
            ScriptStage(llm, settings, MODELS),
        ],
    )


def test_writes_the_script_beat_by_beat(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [HOOK, ENDING]})
    art = build(tmp_path, llm).ensure("creation.script", SCOPE)
    script = art.read_model(SCRIPT_FILE, Script)

    assert (script.id, script.project_id, script.version, script.author) == (
        "scr_t1",
        "prj_t1",
        1,
        "ai",
    )
    assert (script.params.style, script.params.perspective) == ("suspense", "third")
    assert script.params.target_duration_s == 20 and script.params.voice_id == "v1"
    assert [(s.id, s.beat) for s in script.segments] == [
        ("seg_01", "hook"),
        ("seg_02", "ending"),
        ("seg_03", "ending"),
    ]
    assert all(s.kind == "narration" for s in script.segments)
    assert script.segments[2].scene_refs == ["sc_002", "sc_001"]
    assert [(o.beat, o.target_s, o.focus) for o in script.outline] == [
        ("hook", 10, "开场的相遇"),
        ("ending", 10, "分别与结局"),
    ]
    assert art.meta == {
        "segments": 3,
        "chars": 90,
        "target_chars": 90,
        "estimated_s": 20.0,
        "rule_notes": 0,
    }
    assert len(llm.calls) == 2  # one call per beat


def test_every_beat_call_shares_the_same_context_prefix(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [HOOK, ENDING]})
    build(tmp_path, llm).ensure("creation.script", SCOPE)
    first, second = (c[1][0].content for c in llm.calls)

    marker = "现在写第"
    assert first.split(marker)[0] == second.split(marker)[0]  # the cacheable part is identical
    prefix = first.split(marker)[0]
    assert "女孩寻龙" in prefix and "sc_002" in prefix and "悬疑紧凑" in prefix
    assert "台词：我们不能留在这里。" in prefix  # key lines of the scene
    assert "1. hook（10 秒" in prefix and "2. ending" in prefix  # the whole outline
    assert "结局：小龙已死" in prefix and "可以讲出结局" in prefix
    assert all(w in prefix for w in styles.get("suspense").banned_words)

    assert "第 1 节（共 2 节）：hook" in first and "开场的相遇" in first
    assert "这是全文的开头" in first and "这是全文的最后一节" not in first
    assert "第 2 节（共 2 节）：ending" in second and "这是全文的最后一节" in second
    assert "字" * 45 in second  # the beat already written is shown for continuity
    assert "约 45 字" in first and "36–54" in first
    assert llm.calls[0][2].startswith("script_write@") and "+script_beat@" in llm.calls[0][2]


def test_no_spoiler_hides_ending_from_prompt(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [HOOK, ENDING]})
    s = ScriptSettings(target_duration_s=20, voice_id="v1", style=STYLE, spoil_ending=False)
    build(tmp_path, llm, s).ensure("creation.script", SCOPE)
    prompt = llm.calls[0][1][0].content
    assert "小龙已死" not in prompt and "不要透露结局" in prompt


def test_a_beat_that_breaks_the_rules_is_repaired_alone(tmp_path: Path) -> None:
    bad_hook = {"segments": [seg(45, ["sc_404"])]}
    llm = FakeLLM({"script_write": [bad_hook, HOOK, ENDING]})
    art = build(tmp_path, llm).ensure("creation.script", SCOPE)
    assert art.meta["chars"] == 90 and len(llm.calls) == 3
    retry = llm.calls[1][1]
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "sc_404" in retry[2].content
    assert "现在写第 2 节" in llm.calls[2][1][0].content  # the next beat is not affected


def test_a_beat_with_the_wrong_length_is_asked_again(tmp_path: Path) -> None:
    short = {"segments": [seg(20, ["sc_001"])]}
    llm = FakeLLM({"script_write": [short, HOOK, ENDING]})
    build(tmp_path, llm).ensure("creation.script", SCOPE)
    assert "本节共 20 字" in llm.calls[1][1][2].content and "补充" in llm.calls[1][1][2].content


def test_banned_words_are_caught(tmp_path: Path) -> None:
    banned = styles.get("suspense").banned_words[0]
    bad = {"segments": [{"text": "字" * 40 + banned, "scene_refs": ["sc_001"]}]}
    llm = FakeLLM({"script_write": [bad, HOOK, ENDING]})
    build(tmp_path, llm).ensure("creation.script", SCOPE)
    assert f"禁用词：{banned}" in llm.calls[1][1][2].content


def test_gives_up_after_two_repair_rounds(tmp_path: Path) -> None:
    bad = {"segments": [seg(20, ["sc_001"])]}
    llm = FakeLLM({"script_write": [bad, bad, bad, HOOK]})
    with pytest.raises(ScriptError, match=r"beat 'hook'.*补充"):
        build(tmp_path, llm).ensure("creation.script", SCOPE)
    assert len(llm.calls) == 3
    assert not list((tmp_path / "a").glob("creation.script/*/manifest.json"))


def test_llm_error_propagates(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [LLMError("boom")]})
    with pytest.raises(LLMError):
        build(tmp_path, llm).ensure("creation.script", SCOPE)


def test_an_edited_outline_replaces_the_generated_one(tmp_path: Path) -> None:
    edited = ScriptOutline(
        asset_id=ASSET,
        style=STYLE,
        target_duration_s=20,
        beats=[
            OutlineBeat(
                beat="all", scene_refs=["sc_001", "sc_002"], target_s=20, focus="一口气讲完"
            )
        ],
    )
    llm = FakeLLM({"script_write": [{"segments": [seg(45, ["sc_001"]), seg(44, ["sc_002"])]}]})
    settings = ScriptSettings(target_duration_s=20, voice_id="v1", style=STYLE, outline=edited)
    # no outline stage in this graph: the edit must be enough
    engine = Engine(
        ArtifactStore(tmp_path / "a"),
        [StubStory(), StubScenes(), StubTranscript(), ScriptStage(llm, settings, MODELS)],
    )
    script = engine.ensure("creation.script", SCOPE).read_model(SCRIPT_FILE, Script)
    assert [s.beat for s in script.segments] == ["all", "all"]
    assert [o.focus for o in script.outline] == ["一口气讲完"]
    assert "一口气讲完" in llm.calls[0][1][0].content


def test_an_outline_with_unknown_scenes_is_refused(tmp_path: Path) -> None:
    stale = ScriptOutline(
        asset_id=ASSET,
        style=STYLE,
        target_duration_s=20,
        beats=[OutlineBeat(beat="all", scene_refs=["sc_gone"], target_s=20)],
    )
    llm = FakeLLM({"script_write": [HOOK]})
    settings = ScriptSettings(target_duration_s=20, voice_id="v1", style=STYLE, outline=stale)
    engine = Engine(
        ArtifactStore(tmp_path / "a"),
        [StubStory(), StubScenes(), StubTranscript(), ScriptStage(llm, settings, MODELS)],
    )
    with pytest.raises(ScriptError, match="sc_gone"):
        engine.ensure("creation.script", SCOPE)
    assert llm.calls == []


def test_unknown_style_is_a_script_error(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [HOOK]})
    s = ScriptSettings(target_duration_s=20, voice_id="v1", style="neutral")
    with pytest.raises(ScriptError, match="unknown style"):
        build(tmp_path, llm, s).ensure("creation.script", SCOPE)
    assert llm.calls == []


def test_cache_keyed_on_settings_style_and_model(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [HOOK, ENDING] * 4})
    first = build(tmp_path, llm).ensure("creation.script", SCOPE)
    assert build(tmp_path, llm).ensure("creation.script", SCOPE).cache_key == first.cache_key
    assert len(llm.calls) == 2

    other = ScriptSettings(target_duration_s=20, voice_id="v1", style="roast")
    second = build(tmp_path, llm, other).ensure("creation.script", SCOPE)
    assert second.cache_key != first.cache_key and len(llm.calls) == 4

    changed = Engine(
        ArtifactStore(tmp_path / "a"),
        [
            StubStory(),
            StubScenes(),
            StubTranscript(),
            StubOutline(),
            ScriptStage(llm, SETTINGS, {"script_write": "fake/m2"}),
        ],
    )
    assert changed.ensure("creation.script", SCOPE).cache_key != first.cache_key


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        ScriptSettings(target_duration_s=0, voice_id="v")
    with pytest.raises(ValueError):
        ScriptSettings(target_duration_s=10, voice_id="v", chars_per_s=0)


def test_a_name_slip_is_sent_back_for_repair(tmp_path: Path) -> None:
    slip = {"segments": [{"text": "辛特尔" + "字" * 42 + "。", "scene_refs": ["sc_001"]}]}
    llm = FakeLLM({"script_write": [slip, HOOK, ENDING]})
    settings = ScriptSettings(
        target_duration_s=20, voice_id="v1", style=STYLE, names=(KnownName("辛忒尔"),)
    )
    build(tmp_path, llm, settings).ensure("creation.script", SCOPE)
    assert "把人物「辛忒尔」写成了「辛特尔」" in llm.calls[1][1][2].content


def test_confirmed_names_are_part_of_the_cache_key(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [HOOK, ENDING] * 2})
    first = build(tmp_path, llm).ensure("creation.script", SCOPE)
    named = ScriptSettings(
        target_duration_s=20, voice_id="v1", style=STYLE, names=(KnownName("辛忒尔", ["小辛"]),)
    )
    assert build(tmp_path, llm, named).ensure("creation.script", SCOPE).cache_key != first.cache_key


def test_what_the_beats_could_not_fix_is_left_as_rule_annotations(tmp_path: Path) -> None:
    # each beat is at the edge of its own tolerance (+20 %), together they are 20 % over the
    # target, which is more than the whole script may be (15 %)
    long_beats = [{"segments": [seg(54, ["sc_001"])]}, {"segments": [seg(54, ["sc_002"])]}]
    llm = FakeLLM({"script_write": long_beats})
    art = build(tmp_path, llm).ensure("creation.script", SCOPE)
    script = art.read_model(SCRIPT_FILE, Script)
    assert [(a.segment_id, a.type) for a in script.annotations] == [("seg_02", "rule")]
    assert "全文共 108 字" in script.annotations[0].message and art.meta["rule_notes"] == 1
