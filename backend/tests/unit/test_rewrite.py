"""Rewriting one segment: the prompt, the neighbours, the rules and the refusals."""

from __future__ import annotations

from typing import Any

import pytest

from offscreen import styles
from offscreen.domain.index import Act, Scene, Story, TranscriptLine
from offscreen.domain.script import OutlineBeat, Script, ScriptParams, ScriptSegment
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.stages.creation.rewrite import RewriteError, rewrite_segment
from offscreen.stages.creation.script import render_context

PRESET = styles.get("suspense")
STORY = Story(
    asset_id="ast_t1",
    logline="女孩寻龙",
    synopsis="一个女孩为了找回小龙走遍世界。",
    acts=[Act(name="一", scene_ids=["sc_001", "sc_002"], summary="x")],
    ending="小龙已死",
)
SCENES = [
    Scene(id="sc_001", start_ms=0, end_ms=30_000, shot_ids=["sh_1"], summary="相遇"),
    Scene(id="sc_002", start_ms=30_000, end_ms=60_000, shot_ids=["sh_2"], summary="分别"),
]
LINES = [TranscriptLine(id="ln_1", start_ms=1_000, end_ms=3_000, text="我们不能留在这里。")]
OUTLINE = [
    OutlineBeat(beat="hook", scene_refs=["sc_001"], target_s=10, focus="开场"),
    OutlineBeat(beat="ending", scene_refs=["sc_002"], target_s=10, focus="结局"),
]


def seg(
    i: int, chars: int, ref: str = "sc_001", beat: str = "hook", kind: str = "narration"
) -> Any:
    if kind == "original":
        return ScriptSegment(
            id=f"seg_{i:02d}", kind="original", text="（原声）", line_refs=["ln_1"]
        )
    return ScriptSegment(
        id=f"seg_{i:02d}", kind="narration", beat=beat, text="字" * chars + "。", scene_refs=[ref]
    )


SCRIPT = Script(
    id="scr_t1",
    project_id="prj_t1",
    version=1,
    author="ai",
    params=ScriptParams(style="suspense", target_duration_s=20, voice_id="v"),
    outline=OUTLINE,
    segments=[
        seg(1, 20),
        seg(2, 40),
        seg(3, 30, "sc_002", "ending"),
        seg(4, 30, "sc_002", "ending"),
    ],
)


def reply(chars: int, refs: list[str] | None = None) -> dict[str, Any]:
    return {"text": "新" * chars + "。", "scene_refs": refs or ["sc_001"]}


def rewrite(llm: FakeLLM, segment_id: str = "seg_02", instruction: str = "更口语化", **kw: Any):  # type: ignore[no-untyped-def]
    return rewrite_segment(
        llm,
        kw.pop("script", SCRIPT),
        segment_id,
        instruction,
        preset=PRESET,
        story=STORY,
        scenes=SCENES,
        lines=LINES,
        **kw,
    )


def test_returns_the_new_text_and_refs() -> None:
    llm = FakeLLM({"script_rewrite": [reply(38, ["sc_002", "sc_001", "sc_002"])]})
    out = rewrite(llm)
    assert out.text == "新" * 38 + "。" and out.scene_refs == ["sc_002", "sc_001"]


def test_the_prompt_starts_with_the_writing_context_then_the_instruction() -> None:
    llm = FakeLLM({"script_rewrite": [reply(40)]})
    rewrite(llm, instruction="  加一点悬念  ")
    task, messages, version = llm.calls[0]
    prompt = messages[0].content

    context = render_context(PRESET, True, STORY, SCENES, LINES, OUTLINE)
    assert task == "script_rewrite"
    assert prompt.startswith(context.text + "\n\n")  # the same prefix writing used: cacheable
    assert version == f"{context.version}+script_rewrite@1"
    assert "要改写的是 seg_02（节拍：hook，这一节要讲：开场）" in prompt
    assert "原文：" + "字" * 40 + "。" in prompt
    assert "改写要求：加一点悬念" in prompt
    assert "约 40 字" in prompt and "28–52" in prompt


def test_neighbours_are_shown_two_each_way() -> None:
    llm = FakeLLM({"script_rewrite": [reply(30, ["sc_002"])]})
    rewrite(llm, "seg_03")
    prompt = llm.calls[0][1][0].content
    tail = prompt.split("要改写的是")[1]
    assert "它前面的内容" in tail and "字" * 20 in tail and "字" * 40 in tail
    assert (
        "它后面的内容" in tail and tail.count("字" * 30) >= 2
    )  # the next segment and the original
    first = FakeLLM({"script_rewrite": [reply(20)]})
    rewrite(first, "seg_01")
    assert "它前面的内容" not in first.calls[0][1][0].content.split("要改写的是")[1]


def test_rule_violations_are_repaired() -> None:
    llm = FakeLLM({"script_rewrite": [reply(40, ["sc_404"]), reply(10), reply(41)]})
    out = rewrite(llm)
    assert out.text == "新" * 41 + "。" and len(llm.calls) == 3
    assert "sc_404" in llm.calls[1][1][2].content
    assert "太短" in llm.calls[2][1][4].content or "补充" in llm.calls[2][1][4].content


def test_banned_words_are_caught() -> None:
    banned = PRESET.banned_words[0]
    bad = {"text": "新" * 30 + banned, "scene_refs": ["sc_001"]}
    llm = FakeLLM({"script_rewrite": [bad, reply(40)]})
    rewrite(llm)
    assert f"禁用词：{banned}" in llm.calls[1][1][2].content


def test_gives_up_after_two_repair_rounds() -> None:
    llm = FakeLLM({"script_rewrite": [reply(5)] * 3 + [reply(40)]})
    with pytest.raises(RewriteError, match="still breaks the rules"):
        rewrite(llm)
    assert len(llm.calls) == 3


def test_the_length_may_move_within_thirty_percent() -> None:
    llm = FakeLLM({"script_rewrite": [reply(52)]})  # 40 * 1.3
    assert rewrite(llm).text == "新" * 52 + "。"
    too_long = FakeLLM({"script_rewrite": [reply(53), reply(53), reply(53)]})
    with pytest.raises(RewriteError):
        rewrite(too_long)


@pytest.mark.parametrize(
    ("segment_id", "instruction", "message"),
    [
        ("seg_99", "x", "no segment seg_99"),
        ("seg_02", "   ", "instruction is empty"),
        ("seg_02", "字" * 201, "longer than"),
    ],
)
def test_refusals_happen_before_asking_the_model(
    segment_id: str, instruction: str, message: str
) -> None:
    llm = FakeLLM({"script_rewrite": [reply(40)]})
    with pytest.raises(RewriteError, match=message):
        rewrite(llm, segment_id, instruction)
    assert llm.calls == []


def test_original_sound_segments_are_not_rewritten() -> None:
    script = SCRIPT.model_copy(update={"segments": [seg(1, 20), seg(2, 0, kind="original")]})
    llm = FakeLLM({"script_rewrite": [reply(40)]})
    with pytest.raises(RewriteError, match="original-sound"):
        rewrite(llm, "seg_02", script=script)
    assert llm.calls == []
