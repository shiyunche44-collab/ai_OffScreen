"""The fact-check stage: it annotates, it never rewrites, and it ignores what it cannot place."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from offscreen.domain.script import Script, ScriptParams, ScriptReview, ScriptSegment
from offscreen.engine import ArtifactStore, Engine, Stage, StageContext, StageOutput
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.providers.ports import LLMError
from offscreen.stages.creation.review import REVIEW_FILE, ReviewStage
from offscreen.stages.creation.script import SCRIPT_FILE
from offscreen.store.files import write_model

from .test_script import ASSET, SCOPE, StubScenes, StubStory, StubTranscript

MODELS = {"script_critic": "other/critic"}


def make_script(spoil: bool = True) -> Script:
    def seg(i: int, text: str, ref: str) -> ScriptSegment:
        return ScriptSegment(id=f"seg_{i:02d}", kind="narration", text=text, scene_refs=[ref])

    return Script(
        id="scr_t1",
        project_id="prj_t1",
        version=1,
        author="ai",
        params=ScriptParams(
            style="suspense", target_duration_s=20, voice_id="v", spoil_ending=spoil
        ),
        segments=[
            seg(1, "女孩背着龙翻过了三座雪山。", "sc_001"),
            seg(2, "最后小龙死在了她怀里。", "sc_002"),
        ],
    )


class StubScript(Stage):
    name = "creation.script"
    version = 1
    lane = "api"

    def __init__(self, script: Script) -> None:
        self.script = script

    def params(self, scope: Any) -> dict[str, Any]:
        return {"script": self.script.model_dump(mode="json")}

    def run(self, ctx: StageContext) -> StageOutput:
        write_model(ctx.out_dir / SCRIPT_FILE, self.script)
        return StageOutput()


def build(tmp_path: Path, llm: FakeLLM, script: Script | None = None) -> Engine:
    return Engine(
        ArtifactStore(tmp_path / "a"),
        [
            StubStory(),
            StubScenes(),
            StubTranscript(),
            StubScript(script or make_script()),
            ReviewStage(llm, MODELS),
        ],
    )


def finding(segment_id: str, problem: str = "资料里没有这件事") -> dict[str, str]:
    return {"segment_id": segment_id, "problem": problem}


def test_findings_become_fact_check_annotations(tmp_path: Path) -> None:
    llm = FakeLLM(
        {"script_critic": [{"findings": [finding("seg_02", "小龙在场景 sc_002 里没有死")]}]}
    )
    art = build(tmp_path, llm).ensure("creation.review", SCOPE)
    review = art.read_model(REVIEW_FILE, ScriptReview)

    assert review.asset_id == ASSET
    assert [(a.segment_id, a.type, a.message) for a in review.annotations] == [
        ("seg_02", "fact_check", "小龙在场景 sc_002 里没有死")
    ]
    assert art.meta == {"findings": 1, "dropped": 0}


def test_the_prompt_has_the_material_and_the_text_with_ids(tmp_path: Path) -> None:
    llm = FakeLLM({"script_critic": [{"findings": []}]})
    build(tmp_path, llm).ensure("creation.review", SCOPE)
    task, messages, version = llm.calls[0]
    prompt = messages[0].content

    assert task == "script_critic" and version.startswith("script_review@")
    assert "女孩寻龙" in prompt and "结局：小龙已死" in prompt and "sc_002" in prompt
    assert "台词：再见了，小龙。" in prompt
    assert "seg_01（依据 sc_001）女孩背着龙翻过了三座雪山。" in prompt
    assert "你只标注问题，不要改写文案" in prompt
    assert "透露了结局" not in prompt  # this script may spoil


def test_a_no_spoiler_script_is_also_checked_for_leaks(tmp_path: Path) -> None:
    llm = FakeLLM({"script_critic": [{"findings": []}]})
    build(tmp_path, llm, make_script(spoil=False)).ensure("creation.review", SCOPE)
    assert "透露了结局" in llm.calls[0][1][0].content


def test_no_findings_is_a_clean_review(tmp_path: Path) -> None:
    llm = FakeLLM({"script_critic": [{"findings": []}]})
    art = build(tmp_path, llm).ensure("creation.review", SCOPE)
    assert art.read_model(REVIEW_FILE, ScriptReview).annotations == []


def test_findings_about_segments_that_do_not_exist_are_dropped(tmp_path: Path) -> None:
    llm = FakeLLM({"script_critic": [{"findings": [finding("seg_99"), finding("seg_01")]}]})
    art = build(tmp_path, llm).ensure("creation.review", SCOPE)
    assert [a.segment_id for a in art.read_model(REVIEW_FILE, ScriptReview).annotations] == [
        "seg_01"
    ]
    assert art.meta == {"findings": 1, "dropped": 1}


def test_a_very_long_finding_is_cut(tmp_path: Path) -> None:
    llm = FakeLLM({"script_critic": [{"findings": [finding("seg_01", "错" * 1000)]}]})
    art = build(tmp_path, llm).ensure("creation.review", SCOPE)
    message = art.read_model(REVIEW_FILE, ScriptReview).annotations[0].message
    assert len(message) == 300 and message.endswith("…")


def test_the_script_itself_is_untouched(tmp_path: Path) -> None:
    llm = FakeLLM({"script_critic": [{"findings": [finding("seg_01")]}]})
    engine = build(tmp_path, llm)
    before = engine.ensure("creation.script", SCOPE).read_model(SCRIPT_FILE, Script)
    engine.ensure("creation.review", SCOPE)
    assert engine.ensure("creation.script", SCOPE).read_model(SCRIPT_FILE, Script) == before


def test_llm_error_propagates_and_nothing_is_stored(tmp_path: Path) -> None:
    llm = FakeLLM({"script_critic": [LLMError("boom")]})
    with pytest.raises(LLMError):
        build(tmp_path, llm).ensure("creation.review", SCOPE)
    assert not list((tmp_path / "a").glob("creation.review/*/manifest.json"))


def test_cache_follows_the_script_and_the_critic_model(tmp_path: Path) -> None:
    llm = FakeLLM({"script_critic": [{"findings": []}] * 4})
    first = build(tmp_path, llm).ensure("creation.review", SCOPE)
    assert build(tmp_path, llm).ensure("creation.review", SCOPE).cache_key == first.cache_key
    assert len(llm.calls) == 1

    other_script = make_script().model_copy(update={"id": "scr_t2"})
    assert build(tmp_path, llm, other_script).ensure("creation.review", SCOPE).cache_key != (
        first.cache_key
    )
    swapped = Engine(
        ArtifactStore(tmp_path / "a"),
        [
            StubStory(),
            StubScenes(),
            StubTranscript(),
            StubScript(make_script()),
            ReviewStage(llm, {"script_critic": "other/critic2"}),
        ],
    )
    assert swapped.ensure("creation.review", SCOPE).cache_key != first.cache_key
