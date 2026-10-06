"""outline: story + scenes -> outline.json (beats, the scenes each draws on, seconds each).

One LLM call (task `script_outline`) proposes the beats following the style preset's structure
template. Rules check scene refs, names and the total length; violations go back to the model
for at most two repair rounds. The seconds are then fitted deterministically so they add up to
the target exactly. The result can be edited by a person before the text is written."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from offscreen import styles
from offscreen.algo.outline import (
    MIN_BEAT_S,
    OUTLINE_TOLERANCE,
    BeatDraft,
    check_outline,
    fit_durations,
)
from offscreen.algo.story import fmt_clock
from offscreen.domain.index import Scenes, Story
from offscreen.domain.job import Lane
from offscreen.domain.script import OutlineBeat, ScriptOutline
from offscreen.domain.style import StylePreset
from offscreen.engine import ArtifactRef, Scope, Stage, StageContext, StageOutput
from offscreen.prompts import render, template_version
from offscreen.providers.ports import LLM, Message
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.story import STORY_FILE
from offscreen.store.files import write_model

OUTLINE_FILE = "outline.json"
OUTLINE_TASK = "script_outline"
REPAIR_ROUNDS = 2


class OutlineError(RuntimeError):
    pass


@dataclass(frozen=True)
class OutlineSettings:
    target_duration_s: int
    style: str
    spoil_ending: bool = True

    def __post_init__(self) -> None:
        if self.target_duration_s < 1:
            raise ValueError("target_duration_s must be positive")


class BeatReply(BaseModel):
    """One proposed beat (lenient: unknown fields are ignored)."""

    beat: str = Field(min_length=1)
    focus: str = ""
    scene_refs: list[str] = []
    target_s: float = Field(gt=0)


class OutlineReply(BaseModel):
    beats: list[BeatReply] = Field(min_length=1)


class OutlineStage(Stage):
    name = "creation.outline"
    version = 1
    lane: Lane = "api"

    def __init__(
        self, llm: LLM, settings: OutlineSettings, models: Mapping[str, str] | None = None
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.models = dict(models or {})
        """`task -> "provider/model"` for the tasks this stage uses; part of the cache key."""

    def _preset(self) -> StylePreset:
        try:
            return styles.get(self.settings.style)
        except styles.StyleError as e:
            raise OutlineError(str(e)) from e

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.story", scope), ArtifactRef("analysis.scenes", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        cfg = self.settings
        return {
            "asset_id": scope["asset_id"],
            "target_duration_s": cfg.target_duration_s,
            "spoil_ending": cfg.spoil_ending,
            "style": self._preset().model_dump(mode="json"),  # editing a preset invalidates
            "tolerance": OUTLINE_TOLERANCE,
            "min_beat_s": MIN_BEAT_S,
            "language": "zh",
        }

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "models": {OUTLINE_TASK: self.models.get(OUTLINE_TASK)},
            "prompts": {OUTLINE_TASK: template_version("script_outline")},
        }

    def run(self, ctx: StageContext) -> StageOutput:
        cfg = self.settings
        preset = self._preset()
        story = ctx.input("analysis.story").read_model(STORY_FILE, Story)
        scenes = ctx.input("analysis.scenes").read_model(SCENES_FILE, Scenes).scenes
        if not scenes:
            raise OutlineError("no scenes to outline from")
        scene_ids = [s.id for s in scenes]
        if cfg.target_duration_s < MIN_BEAT_S * 2:
            raise OutlineError(f"{cfg.target_duration_s} s is too short for an outline")

        prompt = render(
            "script_outline",
            style_name=preset.name,
            style_description=preset.description,
            tone=preset.tone,
            perspective=preset.perspective,
            target_s=cfg.target_duration_s,
            structure=[
                {
                    "name": b.name,
                    "purpose": b.purpose,
                    "seconds": max(MIN_BEAT_S, round(b.share * cfg.target_duration_s)),
                }
                for b in preset.structure
            ],
            hook_types=preset.hook_types,
            spoil_ending=cfg.spoil_ending,
            logline=story.logline,
            synopsis=story.synopsis,
            turning_points=story.turning_points,
            ending=story.ending,
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
        ctx.progress(0.05, "outlining")
        draft = self._propose(prompt.text, prompt.version, scene_ids, ctx)

        seconds = fit_durations([b.target_s for b in draft.beats], cfg.target_duration_s)
        beats = [
            OutlineBeat(
                beat=b.beat.strip(),
                focus=b.focus.strip(),
                scene_refs=list(dict.fromkeys(b.scene_refs)),
                target_s=s,
            )
            for b, s in zip(draft.beats, seconds, strict=True)
        ]
        outline = ScriptOutline(
            asset_id=ctx.scope["asset_id"],
            style=preset.id,
            target_duration_s=cfg.target_duration_s,
            beats=beats,
        )
        write_model(ctx.out_dir / OUTLINE_FILE, outline)
        ctx.progress(1.0, "done")
        return StageOutput(meta={"beats": len(beats), "total_s": outline.total_s})

    def _propose(
        self, prompt_text: str, prompt_version: str, scene_ids: list[str], ctx: StageContext
    ) -> OutlineReply:
        target = self.settings.target_duration_s
        messages = [Message("user", prompt_text)]
        problems: list[str] = []
        for attempt in range(REPAIR_ROUNDS + 1):
            reply = self.llm.generate(
                OUTLINE_TASK, messages, OutlineReply, prompt_version=prompt_version, max_tokens=4096
            )
            problems = check_outline(
                [BeatDraft(b.beat, b.scene_refs, b.target_s) for b in reply.beats],
                scene_ids,
                target_s=target,
            )
            if not problems:
                return reply
            ctx.progress(0.3 + 0.3 * attempt, f"repairing outline ({len(problems)} problems)")
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
        raise OutlineError("outline still breaks the rules after repair: " + "; ".join(problems))
