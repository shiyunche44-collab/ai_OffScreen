"""script: story + scenes -> script.json (v0: one narration pass, no versioning).

One LLM call (task `script_write`) writes the narration segments, each citing the scenes it
is based on. Deterministic rules then check scene refs, segment lengths and the total length
against the target (`target_duration_s * chars_per_s`); violations go back to the model for
at most two repair rounds. The outline is derived from the written segments. M1 has no
projects or document store, so ids and `version` are fixed: `scr_`/`prj_` + the asset's id
suffix, version 1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from offscreen.algo.script import (
    DEFAULT_CHARS_PER_S,
    LENGTH_TOLERANCE,
    MAX_SEGMENT_CHARS,
    MIN_SEGMENT_CHARS,
    build_outline,
    check_draft,
    count_chars,
    estimate_duration_s,
    target_chars,
)
from offscreen.algo.story import fmt_clock
from offscreen.domain.index import Scenes, Story
from offscreen.domain.job import Lane
from offscreen.domain.script import Script, ScriptParams, ScriptSegment
from offscreen.engine import ArtifactRef, Scope, Stage, StageContext, StageOutput
from offscreen.prompts import render, template_version
from offscreen.providers.ports import LLM, Message
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.story import STORY_FILE
from offscreen.store.files import write_model

SCRIPT_FILE = "script.json"
WRITE_TASK = "script_write"
REPAIR_ROUNDS = 2
SEGMENT_CHARS_HINT = 45
"""Average segment length the prompt aims for; sets the suggested number of segments."""


class ScriptError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScriptSettings:
    target_duration_s: int
    voice_id: str
    style: str = "neutral"
    perspective: Literal["first", "third"] = "third"
    spoil_ending: bool = True
    chars_per_s: float = DEFAULT_CHARS_PER_S

    def __post_init__(self) -> None:
        if self.target_duration_s < 1:
            raise ValueError("target_duration_s must be positive")
        if self.perspective not in ("first", "third"):
            raise ValueError("perspective must be 'first' or 'third'")
        if self.chars_per_s <= 0:
            raise ValueError("chars_per_s must be positive")


class SegmentReply(BaseModel):
    """One written segment (lenient: unknown fields are ignored)."""

    beat: str | None = None
    text: str = Field(min_length=1)
    scene_refs: list[str] = []


class ScriptReply(BaseModel):
    segments: list[SegmentReply] = Field(min_length=1)


class ScriptStage(Stage):
    name = "creation.script"
    version = 1
    lane: Lane = "api"

    def __init__(
        self, llm: LLM, settings: ScriptSettings, models: Mapping[str, str] | None = None
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.models = dict(models or {})
        """`task -> "provider/model"` for the tasks this stage uses; part of the cache key."""

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.story", scope), ArtifactRef("analysis.scenes", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {
            "asset_id": scope["asset_id"],
            **asdict(self.settings),
            "tolerance": LENGTH_TOLERANCE,
            "language": "zh",
        }

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "models": {WRITE_TASK: self.models.get(WRITE_TASK)},
            "prompts": {WRITE_TASK: template_version("script_write")},
        }

    def run(self, ctx: StageContext) -> StageOutput:
        cfg = self.settings
        asset_id = ctx.scope["asset_id"]
        story = ctx.input("analysis.story").read_model(STORY_FILE, Story)
        scenes = ctx.input("analysis.scenes").read_model(SCENES_FILE, Scenes).scenes
        if not scenes:
            raise ScriptError("no scenes to write from")

        target = target_chars(cfg.target_duration_s, cfg.chars_per_s)
        lo, hi = round(target * (1 - LENGTH_TOLERANCE)), round(target * (1 + LENGTH_TOLERANCE))
        prompt = render(
            "script_write",
            style=cfg.style,
            perspective=cfg.perspective,
            spoil_ending=cfg.spoil_ending,
            target_chars=target,
            min_chars=lo,
            max_chars=hi,
            approx_segments=max(1, round(target / SEGMENT_CHARS_HINT)),
            min_seg=MIN_SEGMENT_CHARS * 3,
            max_seg=MAX_SEGMENT_CHARS // 2,
            logline=story.logline,
            synopsis=story.synopsis,
            turning_points=story.turning_points,
            ending=story.ending,
            themes=story.themes,
            scenes=[
                {
                    "id": s.id,
                    "start": fmt_clock(s.start_ms),
                    "end": fmt_clock(s.end_ms),
                    "importance": s.importance,
                    "summary": s.summary,
                }
                for s in scenes
            ],
        )
        ctx.progress(0.05, "writing")
        draft = self._write(prompt.text, prompt.version, [s.id for s in scenes], target, ctx)

        suffix = asset_id.split("_", 1)[1]
        segments = [
            ScriptSegment(
                id=f"seg_{i:02d}",
                kind="narration",
                beat=(seg.beat or "").strip() or None,
                text=seg.text.strip(),
                scene_refs=list(dict.fromkeys(seg.scene_refs)),
            )
            for i, seg in enumerate(draft.segments, 1)
        ]
        script = Script(
            id=f"scr_{suffix}",
            project_id=f"prj_{suffix}",
            version=1,
            author="ai",
            params=ScriptParams(
                style=cfg.style,
                target_duration_s=cfg.target_duration_s,
                perspective=cfg.perspective,
                spoil_ending=cfg.spoil_ending,
                voice_id=cfg.voice_id,
            ),
            outline=build_outline(segments, cfg.chars_per_s),
            segments=segments,
        )
        write_model(ctx.out_dir / SCRIPT_FILE, script)
        chars = sum(count_chars(s.text) for s in segments)
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={
                "segments": len(segments),
                "chars": chars,
                "target_chars": target,
                "estimated_s": round(estimate_duration_s(chars, cfg.chars_per_s), 1),
            }
        )

    def _write(
        self,
        prompt_text: str,
        prompt_version: str,
        scene_ids: list[str],
        target: int,
        ctx: StageContext,
    ) -> ScriptReply:
        messages = [Message("user", prompt_text)]
        problems: list[str] = []
        for attempt in range(REPAIR_ROUNDS + 1):
            reply = self.llm.generate(
                WRITE_TASK, messages, ScriptReply, prompt_version=prompt_version, max_tokens=8192
            )
            problems = check_draft(
                [s.text for s in reply.segments],
                [s.scene_refs for s in reply.segments],
                scene_ids,
                target=target,
            )
            if not problems:
                return reply
            ctx.progress(0.3 + 0.3 * attempt, f"repairing draft ({len(problems)} problems)")
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
        raise ScriptError("script still breaks the rules after repair: " + "; ".join(problems))
