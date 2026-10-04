from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from offscreen.domain.index import Scenes, Shot, Shots, Story, Transcript, TranscriptLine
from offscreen.engine import ArtifactStore, Engine, Scope, Stage, StageContext, StageOutput
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.providers.ports import LLMError
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.story import (
    SCENES_FILE,
    STORY_FILE,
    StoryError,
    StoryStage,
)
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.store.files import write_model

ASSET = "ast_t1"


class StubTranscript(Stage):
    name = "analysis.transcript"
    version = 1
    lane = "cpu"

    def __init__(self, lines: list[TranscriptLine]) -> None:
        self.lines = lines

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"n": len(self.lines), "text": [x.text for x in self.lines]}

    def run(self, ctx: StageContext) -> StageOutput:
        doc = Transcript(asset_id=ASSET, language="en", source="stub", lines=self.lines)
        write_model(ctx.out_dir / TRANSCRIPT_FILE, doc)
        return StageOutput()


class StubShots(Stage):
    name = "analysis.shots"
    version = 1
    lane = "cpu"

    def __init__(self, n: int) -> None:
        self.n = n

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"n": self.n}

    def run(self, ctx: StageContext) -> StageOutput:
        shots = [
            Shot(id=f"sh_{i:04d}", start_ms=i * 10_000, end_ms=(i + 1) * 10_000)
            for i in range(self.n)
        ]
        write_model(ctx.out_dir / SHOTS_FILE, Shots(asset_id=ASSET, shots=shots))
        return StageOutput()


def lines(*items: tuple[int, int, str]) -> list[TranscriptLine]:
    return [
        TranscriptLine(id=f"ln_{i:04d}", start_ms=a, end_ms=b, text=t)
        for i, (a, b, t) in enumerate(items, 1)
    ]


def build(
    tmp_path: Path, llm: FakeLLM, ls: list[TranscriptLine], n_shots: int = 6
) -> tuple[Engine, StoryStage]:
    stage = StoryStage(llm, {"story_chunk": "fake/m", "story": "fake/m"})
    engine = Engine(ArtifactStore(tmp_path / "a"), [StubTranscript(ls), StubShots(n_shots), stage])
    return engine, stage


STORY = {
    "logline": "一个故事",
    "synopsis": "梗概",
    "acts": [{"name": "第一幕", "summary": "开端", "scene_ids": ["sc_001"]}],
    "turning_points": [{"scene_id": "sc_001", "what": "转折"}],
    "ending": "结局",
    "themes": ["成长"],
}
CHUNK = {"summary": "两人相遇", "characters": ["A", "B"], "location": "雪山", "importance": 3}


def test_single_chunk_story_and_scenes(tmp_path: Path) -> None:
    llm = FakeLLM({"story_chunk": [CHUNK], "story": [STORY]})
    engine, _ = build(tmp_path, llm, lines((1000, 2000, "Hello"), (30_000, 31_000, "Bye")))
    art = engine.ensure("analysis.story", {"asset_id": ASSET})

    story = art.read_model(STORY_FILE, Story)
    scenes = art.read_model(SCENES_FILE, Scenes)
    assert story.asset_id == ASSET and story.acts[0].scene_ids == ["sc_001"]
    assert len(scenes.scenes) == 1
    sc = scenes.scenes[0]
    assert (sc.start_ms, sc.end_ms) == (0, 60_000)  # covers the whole film
    assert sc.shot_ids[0] == "sh_0000" and len(sc.shot_ids) == 6
    assert sc.line_ids == ["ln_0001", "ln_0002"]
    assert sc.importance == 1.0  # model said 3: clamped
    assert sc.location == "雪山"
    assert art.meta == {"lines": 2, "scenes": 1, "acts": 1}
    assert [c[0] for c in llm.calls] == ["story_chunk", "story"]
    assert llm.calls[0][2].startswith("story_chunk@") and llm.calls[1][2].startswith("story_merge@")
    assert "[00:01] Hello" in llm.calls[0][1][0].content


def test_multiple_chunks_get_rolling_context_and_contiguous_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("offscreen.stages.analysis.story.MAX_CHUNK_CHARS", 8)

    ls = lines((1000, 2000, "aaaaaaaa"), (25_000, 26_000, "bbbbbbbb"), (50_000, 51_000, "cc"))
    prompts: list[str] = []

    def chunk(_t: str, msgs: Any, _s: Any) -> dict[str, Any]:
        prompts.append(msgs[0].content)
        return {"summary": f"摘要{len(prompts)}"}

    def merge(_t: str, msgs: Any, _s: Any) -> dict[str, Any]:
        assert "sc_001" in msgs[0].content and "摘要3" in msgs[0].content
        return {
            **STORY,
            "acts": [{"name": "全", "summary": "x", "scene_ids": ["sc_001", "sc_002", "sc_003"]}],
        }

    llm = FakeLLM({"story_chunk": chunk, "story": merge})
    engine, _ = build(tmp_path, llm, ls)
    art = engine.ensure("analysis.story", {"asset_id": ASSET})

    scenes = art.read_model(SCENES_FILE, Scenes).scenes
    assert [s.id for s in scenes] == ["sc_001", "sc_002", "sc_003"]
    assert scenes[0].start_ms == 0 and scenes[-1].end_ms == 60_000
    assert all(a.end_ms == b.start_ms for a, b in pairwise(scenes))
    assert sum(len(s.shot_ids) for s in scenes) == 6
    assert "摘要1" in prompts[1] and "摘要1" not in prompts[0]


def test_invalid_scene_refs_get_one_repair_round(tmp_path: Path) -> None:
    bad = {**STORY, "acts": [{"name": "x", "summary": "y", "scene_ids": ["sc_099"]}]}
    llm = FakeLLM({"story_chunk": [CHUNK], "story": [bad, STORY]})
    engine, _ = build(tmp_path, llm, lines((1000, 2000, "Hello")))
    art = engine.ensure("analysis.story", {"asset_id": ASSET})
    assert art.read_model(STORY_FILE, Story).acts[0].scene_ids == ["sc_001"]
    retry = llm.calls[-1][1]
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "sc_099" in retry[2].content and "sc_001" in retry[2].content


def test_invalid_scene_refs_twice_fail(tmp_path: Path) -> None:
    bad = {**STORY, "turning_points": [{"scene_id": "sc_099", "what": "w"}]}
    llm = FakeLLM({"story_chunk": [CHUNK], "story": [bad, bad]})
    engine, _ = build(tmp_path, llm, lines((1000, 2000, "Hello")))
    with pytest.raises(StoryError, match="sc_099"):
        engine.ensure("analysis.story", {"asset_id": ASSET})


def test_no_dialogue_is_an_error(tmp_path: Path) -> None:
    engine, _ = build(tmp_path, FakeLLM({}), [])
    with pytest.raises(StoryError, match="no dialogue"):
        engine.ensure("analysis.story", {"asset_id": ASSET})


def test_llm_failure_propagates_and_leaves_no_artifact(tmp_path: Path) -> None:
    llm = FakeLLM({"story_chunk": [LLMError("boom")]})
    engine, _ = build(tmp_path, llm, lines((1000, 2000, "Hello")))
    with pytest.raises(LLMError):
        engine.ensure("analysis.story", {"asset_id": ASSET})
    assert not list((tmp_path / "a").glob("analysis.story/*/manifest.json"))


def test_second_run_hits_cache_and_model_change_invalidates(tmp_path: Path) -> None:
    llm = FakeLLM({"story_chunk": [CHUNK, CHUNK], "story": [STORY, STORY]})
    ls = lines((1000, 2000, "Hello"))
    engine, _ = build(tmp_path, llm, ls)
    scope = {"asset_id": ASSET}
    first = engine.ensure("analysis.story", scope)
    again = engine.ensure("analysis.story", scope)
    assert again.cache_key == first.cache_key and len(llm.calls) == 2

    other = StoryStage(llm, {"story_chunk": "fake/m2", "story": "fake/m"})
    engine2 = Engine(ArtifactStore(tmp_path / "a"), [StubTranscript(ls), StubShots(6), other])
    third = engine2.ensure("analysis.story", scope)
    assert third.cache_key != first.cache_key and len(llm.calls) == 4
