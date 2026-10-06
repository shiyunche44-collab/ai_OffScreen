"""The outline stage: proposal, repair, exact timing, cache keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from offscreen import styles
from offscreen.domain.index import Act, Scene, Scenes, Story
from offscreen.domain.script import ScriptOutline
from offscreen.engine import ArtifactStore, Engine, Stage, StageContext, StageOutput
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.providers.ports import LLMError
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.story import STORY_FILE
from offscreen.stages.creation.outline import (
    OUTLINE_FILE,
    OutlineError,
    OutlineSettings,
    OutlineStage,
)
from offscreen.store.files import write_model

ASSET = "ast_t1"
SCOPE = {"asset_id": ASSET}
MODELS = {"script_outline": "fake/m"}


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


def beat(name: str, refs: list[str], seconds: float, focus: str = "讲点什么") -> dict[str, Any]:
    return {"beat": name, "focus": focus, "scene_refs": refs, "target_s": seconds}


GOOD = {"beats": [beat("hook", ["sc_001"], 15), beat("ending", ["sc_002", "sc_001"], 45)]}
SETTINGS = OutlineSettings(target_duration_s=60, style="suspense")


def build(tmp_path: Path, llm: FakeLLM, settings: OutlineSettings = SETTINGS) -> Engine:
    return Engine(
        ArtifactStore(tmp_path / "a"),
        [StubStory(), StubScenes(), OutlineStage(llm, settings, MODELS)],
    )


def test_writes_a_valid_outline(tmp_path: Path) -> None:
    llm = FakeLLM({"script_outline": [GOOD]})
    art = build(tmp_path, llm).ensure("creation.outline", SCOPE)
    outline = art.read_model(OUTLINE_FILE, ScriptOutline)

    assert (outline.asset_id, outline.style, outline.target_duration_s) == (ASSET, "suspense", 60)
    assert [(b.beat, b.target_s, b.scene_refs) for b in outline.beats] == [
        ("hook", 15, ["sc_001"]),
        ("ending", 45, ["sc_002", "sc_001"]),
    ]
    assert outline.beats[0].focus == "讲点什么"
    assert art.meta == {"beats": 2, "total_s": 60}


def test_prompt_carries_the_style_the_story_and_the_target(tmp_path: Path) -> None:
    llm = FakeLLM({"script_outline": [GOOD]})
    build(tmp_path, llm).ensure("creation.outline", SCOPE)
    prompt = llm.calls[0][1][0].content
    preset = styles.get("suspense")
    assert preset.name in prompt and preset.tone in prompt
    assert all(b.name in prompt and b.purpose in prompt for b in preset.structure)
    assert "解说总时长：60 秒" in prompt and "女孩寻龙" in prompt and "sc_002" in prompt
    assert "结局：小龙已死" in prompt and "可以讲出结局" in prompt
    assert llm.calls[0][2].startswith("script_outline@")


def test_no_spoiler_hides_the_ending(tmp_path: Path) -> None:
    llm = FakeLLM({"script_outline": [GOOD]})
    s = OutlineSettings(target_duration_s=60, style="suspense", spoil_ending=False)
    build(tmp_path, llm, s).ensure("creation.outline", SCOPE)
    prompt = llm.calls[0][1][0].content
    assert "小龙已死" not in prompt and "不要透露结局" in prompt


def test_seconds_are_fitted_to_the_target_exactly(tmp_path: Path) -> None:
    # the model's numbers add up to 66 s (within tolerance); the outline must add up to 60 s
    llm = FakeLLM(
        {"script_outline": [{"beats": [beat("a", ["sc_001"], 30), beat("b", ["sc_002"], 36)]}]}
    )
    outline = (
        build(tmp_path, llm)
        .ensure("creation.outline", SCOPE)
        .read_model(OUTLINE_FILE, ScriptOutline)
    )
    assert outline.total_s == 60
    assert outline.beats[1].target_s > outline.beats[0].target_s


def test_rule_violations_are_repaired(tmp_path: Path) -> None:
    bad = {"beats": [beat("hook", ["sc_404"], 15), beat("ending", [], 45)]}
    llm = FakeLLM({"script_outline": [bad, GOOD]})
    art = build(tmp_path, llm).ensure("creation.outline", SCOPE)
    assert art.meta["beats"] == 2
    retry = llm.calls[-1][1]
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "sc_404" in retry[2].content and "没有 scene_refs" in retry[2].content


def test_gives_up_after_two_repair_rounds(tmp_path: Path) -> None:
    bad = {"beats": [beat("hook", ["sc_404"], 60)]}
    llm = FakeLLM({"script_outline": [bad, bad, bad, GOOD]})
    with pytest.raises(OutlineError, match="sc_404"):
        build(tmp_path, llm).ensure("creation.outline", SCOPE)
    assert len(llm.calls) == 3
    assert not list((tmp_path / "a").glob("creation.outline/*/manifest.json"))


def test_llm_error_propagates(tmp_path: Path) -> None:
    llm = FakeLLM({"script_outline": [LLMError("boom")]})
    with pytest.raises(LLMError):
        build(tmp_path, llm).ensure("creation.outline", SCOPE)


def test_unknown_style_is_an_outline_error(tmp_path: Path) -> None:
    llm = FakeLLM({"script_outline": [GOOD]})
    with pytest.raises(OutlineError, match="unknown style"):
        build(tmp_path, llm, OutlineSettings(60, "neutral")).ensure("creation.outline", SCOPE)
    assert llm.calls == []


def test_too_short_a_target_is_refused_before_asking_the_model(tmp_path: Path) -> None:
    llm = FakeLLM({"script_outline": [GOOD]})
    with pytest.raises(OutlineError, match="too short"):
        build(tmp_path, llm, OutlineSettings(5, "suspense")).ensure("creation.outline", SCOPE)
    assert llm.calls == []


def test_cache_is_keyed_on_style_target_and_model(tmp_path: Path) -> None:
    llm = FakeLLM({"script_outline": [GOOD, GOOD, GOOD, GOOD]})
    first = build(tmp_path, llm).ensure("creation.outline", SCOPE)
    assert build(tmp_path, llm).ensure("creation.outline", SCOPE).cache_key == first.cache_key
    assert len(llm.calls) == 1

    other_style = build(tmp_path, llm, OutlineSettings(60, "roast")).ensure(
        "creation.outline", SCOPE
    )
    assert other_style.cache_key != first.cache_key
    other_len = build(tmp_path, llm, OutlineSettings(66, "suspense")).ensure(
        "creation.outline", SCOPE
    )
    assert other_len.cache_key not in {first.cache_key, other_style.cache_key}
    changed = Engine(
        ArtifactStore(tmp_path / "a"),
        [StubStory(), StubScenes(), OutlineStage(llm, SETTINGS, {"script_outline": "fake/m2"})],
    )
    assert changed.ensure("creation.outline", SCOPE).cache_key != first.cache_key


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        OutlineSettings(target_duration_s=0, style="suspense")
