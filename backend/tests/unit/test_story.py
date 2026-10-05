"""The story stage: acts and turning points from scenes, then the whole story, all anchored."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from offscreen.domain.index import Scene, Scenes, Story
from offscreen.engine import (
    ArtifactStore,
    Engine,
    Scope,
    Stage,
    StageCanceled,
    StageContext,
    StageOutput,
)
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.providers.ports import LLMError
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.story import STORY_FILE, StoryError, StoryStage
from offscreen.store.files import write_model

ASSET = "ast_t1"
SCOPE = {"asset_id": ASSET}
IDS = [f"sc_{i:03d}" for i in range(1, 7)]


class StubScenes(Stage):
    name = "analysis.scenes"
    version = 1
    lane = "api"

    def __init__(self, n: int = 6, tag: str = "a") -> None:
        self.n, self.tag = n, tag

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"n": self.n, "tag": self.tag}

    def run(self, ctx: StageContext) -> StageOutput:
        scenes = [
            Scene(
                id=f"sc_{i:03d}",
                start_ms=(i - 1) * 60_000,
                end_ms=i * 60_000,
                shot_ids=[f"sh_{i:04d}"],
                summary=f"场景 {i} 发生的事" + ("" if self.tag == "a" else self.tag),
                location="雪山" if i % 2 else None,
                importance=0.1 * i,
            )
            for i in range(1, self.n + 1)
        ]
        write_model(ctx.out_dir / SCENES_FILE, Scenes(asset_id=ASSET, scenes=scenes))
        return StageOutput()


ACTS = {
    "acts": [
        {"name": "开端", "summary": "相遇", "scene_ids": IDS[:2]},
        {"name": "发展", "summary": "旅途", "scene_ids": IDS[2:5]},
        {"name": "结局", "summary": "告别", "scene_ids": IDS[5:]},
    ],
    "turning_points": [{"scene_id": "sc_003", "what": "找到龙"}],
}
SYNTHESIS = {
    "logline": "一个女孩寻龙",
    "synopsis": "梗概" * 50,
    "ending": "她独自离开",
    "ending_scene_ids": ["sc_006"],
    "themes": ["陪伴", "失去"],
}


def build(tmp_path: Path, llm: FakeLLM, n: int = 6, models: dict[str, str] | None = None) -> Engine:
    stage = StoryStage(llm, models or {"story": "fake/m"})
    return Engine(ArtifactStore(tmp_path / "a"), [StubScenes(n), stage])


def test_story_is_built_in_two_levels_and_anchored_in_scenes(tmp_path: Path) -> None:
    llm = FakeLLM({"story": [ACTS, SYNTHESIS]})
    art = build(tmp_path, llm).ensure("analysis.story", SCOPE)
    story = art.read_model(STORY_FILE, Story)

    assert story.logline == "一个女孩寻龙" and story.themes == ["陪伴", "失去"]
    assert [(a.name, a.scene_ids) for a in story.acts] == [
        (a["name"], a["scene_ids"]) for a in ACTS["acts"]
    ]
    assert [(t.scene_id, t.what) for t in story.turning_points] == [("sc_003", "找到龙")]
    assert (story.ending, story.ending_scene_ids) == ("她独自离开", ["sc_006"])
    assert story.relations == []  # no characters yet
    assert art.meta == {"scenes": 6, "acts": 3, "turning_points": 1}
    assert [c[0] for c in llm.calls] == ["story", "story"]
    assert llm.calls[0][2].startswith("story_acts@") and llm.calls[1][2].startswith(
        "story_synthesis@"
    )


def test_prompts_carry_the_scene_summaries_then_the_acts(tmp_path: Path) -> None:
    llm = FakeLLM({"story": [ACTS, SYNTHESIS]})
    build(tmp_path, llm).ensure("analysis.story", SCOPE)
    acts_prompt = llm.calls[0][1][0].content
    assert "sc_001（00:00 – 01:00，地点：雪山，重要度 0.1）" in acts_prompt
    assert (
        "场景 6 发生的事" in acts_prompt
        and "地点：" not in acts_prompt.split("sc_002")[1].split("sc_003")[0]
    )
    synthesis_prompt = llm.calls[1][1][0].content
    assert "第 1 幕「开端」（00:00 – 02:00，场景 sc_001、sc_002）" in synthesis_prompt
    assert "- sc_003：找到龙" in synthesis_prompt


def test_acts_must_cover_every_scene_once_and_get_one_repair_round(tmp_path: Path) -> None:
    bad = {**ACTS, "acts": [ACTS["acts"][0], ACTS["acts"][2]]}  # scenes 3-5 are in no act
    llm = FakeLLM({"story": [bad, ACTS, SYNTHESIS]})
    art = build(tmp_path, llm).ensure("analysis.story", SCOPE)
    assert art.meta["acts"] == 3
    repair = llm.calls[1][1]
    assert (
        repair[-1].role == "user"
        and "这些场景不属于任何一幕：sc_003、sc_004、sc_005" in repair[-1].content
    )
    assert "sc_001, sc_002" in repair[-1].content  # the valid ids are listed again


def test_a_scene_in_two_acts_or_an_unknown_one_is_refused(tmp_path: Path) -> None:
    twice = {
        "acts": [
            {"name": "a", "summary": "x", "scene_ids": IDS},
            {"name": "b", "summary": "y", "scene_ids": ["sc_002", "sc_099"]},
        ]
    }
    llm = FakeLLM({"story": [twice, twice]})
    with pytest.raises(StoryError, match="sc_002") as e:
        build(tmp_path, llm).ensure("analysis.story", SCOPE)
    assert "sc_099" in str(e.value) and len(llm.calls) == 2  # one repair round, then give up


def test_a_turning_point_must_cite_an_existing_scene(tmp_path: Path) -> None:
    bad = {**ACTS, "turning_points": [{"scene_id": "sc_404", "what": "x"}]}
    llm = FakeLLM({"story": [bad, bad]})
    with pytest.raises(StoryError, match="sc_404"):
        build(tmp_path, llm).ensure("analysis.story", SCOPE)


def test_the_ending_must_be_anchored_too(tmp_path: Path) -> None:
    unanchored = {**SYNTHESIS, "ending_scene_ids": []}
    llm = FakeLLM({"story": [ACTS, unanchored, SYNTHESIS]})
    art = build(tmp_path, llm).ensure("analysis.story", SCOPE)
    assert art.read_model(STORY_FILE, Story).ending_scene_ids == ["sc_006"]
    assert "ending_scene_ids" in llm.calls[2][1][-1].content

    llm2 = FakeLLM(
        {
            "story": [
                ACTS,
                {**SYNTHESIS, "ending_scene_ids": ["sc_77"]},
                {**SYNTHESIS, "ending_scene_ids": ["sc_77"]},
            ]
        }
    )
    with pytest.raises(StoryError, match="结局引用了不存在的场景 sc_77"):
        build(tmp_path / "x", llm2).ensure("analysis.story", SCOPE)


def test_a_story_without_an_ending_needs_no_anchor(tmp_path: Path) -> None:
    llm = FakeLLM({"story": [ACTS, {**SYNTHESIS, "ending": None, "ending_scene_ids": []}]})
    story = build(tmp_path, llm).ensure("analysis.story", SCOPE).read_model(STORY_FILE, Story)
    assert story.ending is None and story.ending_scene_ids == []


def test_no_scenes_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(StoryError, match="no scenes"):
        build(tmp_path, FakeLLM({}), n=0).ensure("analysis.story", SCOPE)


def test_a_model_failure_propagates_and_leaves_no_artifact(tmp_path: Path) -> None:
    llm = FakeLLM({"story": [LLMError("boom")]})
    with pytest.raises(LLMError):
        build(tmp_path, llm).ensure("analysis.story", SCOPE)
    assert list((tmp_path / "a" / "analysis.story").iterdir()) == []


def test_cancel_between_the_two_levels(tmp_path: Path) -> None:
    llm = FakeLLM({"story": [ACTS, SYNTHESIS]})
    flag = {"stop": False}

    def progress(stage: str, _f: float, msg: str) -> None:
        if stage == "analysis.story" and msg.startswith("acts"):
            flag["stop"] = True

    engine = Engine(
        ArtifactStore(tmp_path / "a"),
        [StubScenes(), StoryStage(llm, {"story": "fake/m"})],
        progress=progress,
        is_canceled=lambda: flag["stop"],
    )
    with pytest.raises(StageCanceled):
        engine.ensure("analysis.story", SCOPE)
    assert len(llm.calls) == 1  # the second level was never asked


def test_a_second_run_hits_the_cache_and_a_new_model_or_scenes_invalidate_it(
    tmp_path: Path,
) -> None:
    llm = FakeLLM({"story": [ACTS, SYNTHESIS, ACTS, SYNTHESIS, ACTS, SYNTHESIS]})
    first = build(tmp_path, llm).ensure("analysis.story", SCOPE)
    again = build(tmp_path, llm).ensure("analysis.story", SCOPE)
    assert again.cache_key == first.cache_key and len(llm.calls) == 2

    other_model = build(tmp_path, llm, models={"story": "fake/other"}).ensure(
        "analysis.story", SCOPE
    )
    assert other_model.cache_key != first.cache_key and len(llm.calls) == 4

    changed_scenes = Engine(
        ArtifactStore(tmp_path / "a"), [StubScenes(6, "b"), StoryStage(llm, {"story": "fake/m"})]
    )
    assert changed_scenes.ensure("analysis.story", SCOPE).cache_key != first.cache_key
