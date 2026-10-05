from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from offscreen.domain.index import Act, Scene, Scenes, Story
from offscreen.domain.script import Script
from offscreen.engine import ArtifactStore, Engine, Stage, StageContext, StageOutput
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.providers.ports import LLMError
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.story import STORY_FILE
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


def seg(chars: int, refs: list[str], beat: str | None = None) -> dict[str, Any]:
    return {"beat": beat, "text": "字" * chars + "。", "scene_refs": refs}


# 20 s * 4.5 = 90 chars target, allowed 77-104.
GOOD = {"segments": [seg(45, ["sc_001"], "hook"), seg(45, ["sc_002", "sc_001"], "ending")]}
SETTINGS = ScriptSettings(target_duration_s=20, voice_id="v1")


def build(tmp_path: Path, llm: FakeLLM, settings: ScriptSettings = SETTINGS) -> Engine:
    return Engine(
        ArtifactStore(tmp_path / "a"),
        [StubStory(), StubScenes(), ScriptStage(llm, settings, MODELS)],
    )


def test_writes_valid_script_document(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [GOOD]})
    art = build(tmp_path, llm).ensure("creation.script", SCOPE)
    script = art.read_model(SCRIPT_FILE, Script)

    assert (script.id, script.project_id, script.version, script.author) == (
        "scr_t1",
        "prj_t1",
        1,
        "ai",
    )
    assert script.params.target_duration_s == 20 and script.params.voice_id == "v1"
    assert [s.id for s in script.segments] == ["seg_01", "seg_02"]
    assert all(s.kind == "narration" for s in script.segments)
    assert script.segments[1].scene_refs == ["sc_002", "sc_001"]
    assert [(o.beat, o.target_s) for o in script.outline] == [("hook", 10), ("ending", 10)]
    assert art.meta == {"segments": 2, "chars": 90, "target_chars": 90, "estimated_s": 20.0}

    prompt = llm.calls[0][1][0].content
    assert "约 90 字" in prompt and "女孩寻龙" in prompt and "sc_002" in prompt
    assert "结局：小龙已死" in prompt and "可以讲出结局" in prompt
    assert llm.calls[0][2].startswith("script_write@")


def test_no_spoiler_hides_ending_from_prompt(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [GOOD]})
    s = ScriptSettings(target_duration_s=20, voice_id="v1", spoil_ending=False)
    build(tmp_path, llm, s).ensure("creation.script", SCOPE)
    prompt = llm.calls[0][1][0].content
    assert "小龙已死" not in prompt and "不要透露结局" in prompt


def test_rule_violations_are_repaired(tmp_path: Path) -> None:
    short = {"segments": [seg(20, ["sc_001"]), seg(20, ["sc_404"])]}
    llm = FakeLLM({"script_write": [short, GOOD]})
    art = build(tmp_path, llm).ensure("creation.script", SCOPE)
    assert art.meta["chars"] == 90
    retry = llm.calls[-1][1]
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "sc_404" in retry[2].content and "补充" in retry[2].content


def test_gives_up_after_two_repair_rounds(tmp_path: Path) -> None:
    bad = {"segments": [seg(20, ["sc_001"])]}
    llm = FakeLLM({"script_write": [bad, bad, bad, GOOD]})
    with pytest.raises(ScriptError, match="补充"):
        build(tmp_path, llm).ensure("creation.script", SCOPE)
    assert len(llm.calls) == 3
    assert not list((tmp_path / "a").glob("creation.script/*/manifest.json"))


def test_llm_error_propagates(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [LLMError("boom")]})
    with pytest.raises(LLMError):
        build(tmp_path, llm).ensure("creation.script", SCOPE)


def test_cache_keyed_on_settings_and_model(tmp_path: Path) -> None:
    llm = FakeLLM({"script_write": [GOOD, GOOD, GOOD]})
    first = build(tmp_path, llm).ensure("creation.script", SCOPE)
    assert build(tmp_path, llm).ensure("creation.script", SCOPE).cache_key == first.cache_key
    assert len(llm.calls) == 1

    other = ScriptSettings(target_duration_s=20, voice_id="v1", style="suspense")
    second = build(tmp_path, llm, other).ensure("creation.script", SCOPE)
    assert second.cache_key != first.cache_key and len(llm.calls) == 2

    changed = Engine(
        ArtifactStore(tmp_path / "a"),
        [StubStory(), StubScenes(), ScriptStage(llm, SETTINGS, {"script_write": "fake/m2"})],
    )
    assert changed.ensure("creation.script", SCOPE).cache_key != first.cache_key


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        ScriptSettings(target_duration_s=0, voice_id="v")
    with pytest.raises(ValueError):
        ScriptSettings(target_duration_s=10, voice_id="v", chars_per_s=0)
    with pytest.raises(ValueError):
        ScriptSettings(target_duration_s=10, voice_id="v", perspective="second")  # type: ignore[arg-type]
