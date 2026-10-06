"""Rewriting one segment of a script on request ("more colloquial", "add suspense").

Not a cached stage: the result becomes a new version of the script document, written by the
job service. The prompt starts with the same context as writing the script (see
`script.render_context`), so the provider's prompt cache is reused; only the instructions at the
end differ. The new text goes through the same rules as written text (scene refs, length, banned
words, names) with up to two repair rounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from offscreen.algo.script import DEFAULT_CHARS_PER_S, count_chars
from offscreen.algo.script_rules import (
    MAX_SEGMENT_CHARS,
    MIN_SEGMENT_CHARS,
    KnownName,
    Rules,
    SegmentDraft,
    check,
)
from offscreen.domain.index import Scene, Story, TranscriptLine
from offscreen.domain.script import Script
from offscreen.domain.style import StylePreset
from offscreen.prompts import render
from offscreen.providers.ports import LLM, Message
from offscreen.stages.creation.script import render_context

REWRITE_TASK = "script_rewrite"
REPAIR_ROUNDS = 2
NEIGHBOURS = 2
"""Segments shown before and after the one being rewritten."""
LENGTH_TOLERANCE = 0.3
"""A rewrite may ask for a longer or shorter text ("add suspense"), within this."""
MAX_INSTRUCTION_CHARS = 200


class RewriteError(RuntimeError):
    pass


class RewriteReply(BaseModel):
    text: str = Field(min_length=1)
    scene_refs: list[str] = []


@dataclass(frozen=True)
class Rewritten:
    text: str
    scene_refs: list[str]


def rewrite_segment(
    llm: LLM,
    script: Script,
    segment_id: str,
    instruction: str,
    *,
    preset: StylePreset,
    story: Story,
    scenes: Sequence[Scene],
    lines: Sequence[TranscriptLine],
    names: Sequence[KnownName] = (),
    chars_per_s: float = DEFAULT_CHARS_PER_S,
) -> Rewritten:
    instruction = instruction.strip()
    if not instruction:
        raise RewriteError("the instruction is empty")
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        raise RewriteError(f"the instruction is longer than {MAX_INSTRUCTION_CHARS} characters")
    index = next((i for i, s in enumerate(script.segments) if s.id == segment_id), None)
    if index is None:
        raise RewriteError(f"the script has no segment {segment_id}")
    segment = script.segments[index]
    if segment.kind != "narration":
        raise RewriteError(
            f"{segment_id} is an original-sound segment; only narration is rewritten"
        )

    context = render_context(
        preset, script.params.spoil_ending, story, scenes, lines, script.outline
    )
    beat = next((b for b in script.outline if b.beat == segment.beat), None)
    before = [s.text for s in script.segments[max(0, index - NEIGHBOURS) : index]]
    after = [s.text for s in script.segments[index + 1 : index + 1 + NEIGHBOURS]]
    target = max(1, count_chars(segment.text))
    lo, hi = round(target * (1 - LENGTH_TOLERANCE)), round(target * (1 + LENGTH_TOLERANCE))
    scene_ids = [s.id for s in scenes]
    rules = Rules(
        scene_ids=scene_ids,
        target_chars=target,
        tolerance=LENGTH_TOLERANCE,
        min_segment_chars=min(MIN_SEGMENT_CHARS, lo),
        max_segment_chars=max(hi, MAX_SEGMENT_CHARS),
        banned_words=preset.banned_words,
        names=names,
        whole="这一段",
    )
    instructions = render(
        "script_rewrite",
        segment_id=segment_id,
        beat=segment.beat,
        focus=beat.focus if beat else "",
        before=before,
        after=after,
        text=segment.text,
        scene_refs=segment.scene_refs,
        instruction=instruction,
        target_chars=target,
        min_chars=lo,
        max_chars=hi,
    )
    messages = [Message("user", f"{context.text}\n\n{instructions.text}")]
    version = f"{context.version}+{instructions.version}"

    problems: list[str] = []
    for _ in range(REPAIR_ROUNDS + 1):
        reply = llm.generate(
            REWRITE_TASK, messages, RewriteReply, prompt_version=version, max_tokens=2048
        )
        problems = [v.message for v in check([SegmentDraft(reply.text, reply.scene_refs)], rules)]
        if not problems:
            return Rewritten(reply.text.strip(), list(dict.fromkeys(reply.scene_refs)))
        messages = [
            *messages,
            Message("assistant", reply.model_dump_json()),
            Message(
                "user",
                "上一稿有以下问题：\n- "
                + "\n- ".join(problems)
                + f"\n场景编号只能用：{', '.join(scene_ids)}。请修改后重新输出完整 JSON。",
            ),
        ]
    raise RewriteError("the rewrite still breaks the rules after repair: " + "; ".join(problems))
