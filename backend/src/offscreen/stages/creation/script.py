"""script: outline + story + scenes -> script.json, written beat by beat.

The outline (M4-04) says which beats the text has, which scenes each draws on and how many
seconds each gets. A person's edit of it, when there is one, comes in through the settings and
replaces the generated one. One LLM call (task `script_write`) writes each beat. Every call
starts with the same context (style, story, scenes with their key lines, the whole outline) and
differs only in the beat instructions at the end, so the provider's prompt cache is reused
across beats and, later, across single-segment rewrites.

Deterministic rules check each beat (scene refs, segment lengths, banned words, length against
the beat's share of the target); violations go back to the model for at most two repair rounds.
The artifact has fixed ids (`scr_`/`prj_` + the asset's id suffix, version 1); the job service
stores it in the project's document history as a new version."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from offscreen import styles
from offscreen.algo.script import (
    BEAT_TOLERANCE,
    DEFAULT_CHARS_PER_S,
    count_chars,
    estimate_duration_s,
    key_lines,
    scene_lines,
    target_chars,
)
from offscreen.algo.script_rules import (
    LENGTH_TOLERANCE,
    MIN_SEGMENT_CHARS,
    KnownName,
    Rules,
    SegmentDraft,
    as_annotations,
    check,
)
from offscreen.algo.story import fmt_clock
from offscreen.domain.index import Scenes, Story, Transcript
from offscreen.domain.job import Lane
from offscreen.domain.script import OutlineBeat, Script, ScriptOutline, ScriptParams, ScriptSegment
from offscreen.domain.style import StylePreset
from offscreen.engine import ArtifactRef, Scope, Stage, StageContext, StageOutput
from offscreen.prompts import render, template_version
from offscreen.providers.ports import LLM, Message
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.story import STORY_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.stages.creation.outline import OUTLINE_FILE
from offscreen.store.files import write_model

SCRIPT_FILE = "script.json"
WRITE_TASK = "script_write"
REPAIR_ROUNDS = 2
SEGMENT_CHARS_HINT = 45
"""Average segment length the prompt aims for; sets the suggested number of segments."""
PREVIOUS_SEGMENTS = 3
"""How many of the segments already written the next beat is shown, for continuity."""


class ScriptError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScriptSettings:
    target_duration_s: int
    voice_id: str
    style: str = "suspense"
    """A style preset id (`offscreen style list`)."""
    spoil_ending: bool = True
    chars_per_s: float = DEFAULT_CHARS_PER_S
    outline: ScriptOutline | None = None
    """A person's edit of the outline; None: use the generated one (`creation.outline`)."""
    names: tuple[KnownName, ...] = ()
    """Confirmed characters; the text must name them as given (rule `name`)."""

    def __post_init__(self) -> None:
        if self.target_duration_s < 1:
            raise ValueError("target_duration_s must be positive")
        if self.chars_per_s <= 0:
            raise ValueError("chars_per_s must be positive")


class SegmentReply(BaseModel):
    """One written segment (lenient: unknown fields are ignored)."""

    text: str = Field(min_length=1)
    scene_refs: list[str] = []


class BeatReply(BaseModel):
    segments: list[SegmentReply] = Field(min_length=1)


class ScriptStage(Stage):
    name = "creation.script"
    version = 2
    lane: Lane = "api"

    def __init__(
        self, llm: LLM, settings: ScriptSettings, models: Mapping[str, str] | None = None
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.models = dict(models or {})
        """`task -> "provider/model"` for the tasks this stage uses; part of the cache key."""

    def _preset(self) -> StylePreset:
        try:
            return styles.get(self.settings.style)
        except styles.StyleError as e:
            raise ScriptError(str(e)) from e

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        refs = [
            ArtifactRef("analysis.story", scope),
            ArtifactRef("analysis.scenes", scope),
            ArtifactRef("analysis.transcript", scope),
        ]
        if self.settings.outline is None:
            refs.append(ArtifactRef("creation.outline", scope))
        return refs

    def params(self, scope: Scope) -> dict[str, Any]:
        cfg = self.settings
        return {
            "asset_id": scope["asset_id"],
            "target_duration_s": cfg.target_duration_s,
            "voice_id": cfg.voice_id,
            "style": self._preset().model_dump(mode="json"),  # editing a preset invalidates
            "spoil_ending": cfg.spoil_ending,
            "chars_per_s": cfg.chars_per_s,
            "outline": cfg.outline.model_dump(mode="json") if cfg.outline else None,
            "tolerance": BEAT_TOLERANCE,
            "names": [{"name": k.name, "aliases": list(k.aliases)} for k in cfg.names],
            "language": "zh",
        }

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "models": {WRITE_TASK: self.models.get(WRITE_TASK)},
            "prompts": {
                WRITE_TASK: template_version("script_write"),
                "script_beat": template_version("script_beat"),
            },
        }

    def run(self, ctx: StageContext) -> StageOutput:
        cfg = self.settings
        preset = self._preset()
        asset_id = ctx.scope["asset_id"]
        story = ctx.input("analysis.story").read_model(STORY_FILE, Story)
        scenes = ctx.input("analysis.scenes").read_model(SCENES_FILE, Scenes).scenes
        lines = ctx.input("analysis.transcript").read_model(TRANSCRIPT_FILE, Transcript).lines
        if not scenes:
            raise ScriptError("no scenes to write from")
        outline = cfg.outline or ctx.input("creation.outline").read_model(
            OUTLINE_FILE, ScriptOutline
        )
        scene_ids = [s.id for s in scenes]
        unknown = sorted({r for b in outline.beats for r in b.scene_refs} - set(scene_ids))
        if unknown:
            raise ScriptError(
                f"the outline refers to scenes the movie does not have: {', '.join(unknown)}"
            )

        context = render(
            "script_write",
            style_name=preset.name,
            style_description=preset.description,
            tone=preset.tone,
            perspective=preset.perspective,
            phrases=preset.phrases,
            banned_words=preset.banned_words,
            spoil_ending=cfg.spoil_ending,
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
                    "lines": key_lines(scene_lines(s, lines)),
                }
                for s in scenes
            ],
            outline=[b.model_dump() for b in outline.beats],
            total_s=outline.total_s,
        )

        segments: list[ScriptSegment] = []
        for n, beat in enumerate(outline.beats, 1):
            ctx.progress(0.9 * (n - 1) / len(outline.beats), f"writing {beat.beat}")
            written = self._write_beat(
                context.text, context.version, preset, outline, n, beat, segments, scene_ids, ctx
            )
            first = len(segments) + 1
            segments.extend(
                ScriptSegment(
                    id=f"seg_{first + i:02d}",
                    kind="narration",
                    beat=beat.beat,
                    text=seg.text.strip(),
                    scene_refs=list(dict.fromkeys(seg.scene_refs)),
                )
                for i, seg in enumerate(written)
            )

        shortest_beat_lo = min(
            round(target_chars(b.target_s, cfg.chars_per_s) * (1 - BEAT_TOLERANCE))
            for b in outline.beats
        )
        suffix = asset_id.split("_", 1)[1]
        script = Script(
            id=f"scr_{suffix}",
            project_id=f"prj_{suffix}",
            version=1,
            author="ai",
            params=ScriptParams(
                style=preset.id,
                target_duration_s=cfg.target_duration_s,
                perspective=preset.perspective,
                spoil_ending=cfg.spoil_ending,
                voice_id=cfg.voice_id,
            ),
            outline=outline.beats,
            segments=segments,
            annotations=as_annotations(
                check(
                    [SegmentDraft(g.text, g.scene_refs, g.id) for g in segments],
                    Rules(
                        scene_ids=scene_ids,
                        target_chars=target_chars(outline.total_s, cfg.chars_per_s),
                        tolerance=LENGTH_TOLERANCE,
                        min_segment_chars=min(MIN_SEGMENT_CHARS, shortest_beat_lo),
                        banned_words=preset.banned_words,
                        names=cfg.names,
                    ),
                ),
                [g.id for g in segments],
            ),
        )
        write_model(ctx.out_dir / SCRIPT_FILE, script)
        chars = sum(count_chars(s.text) for s in segments)
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={
                "segments": len(segments),
                "chars": chars,
                "target_chars": target_chars(outline.total_s, cfg.chars_per_s),
                "estimated_s": round(estimate_duration_s(chars, cfg.chars_per_s), 1),
                "rule_notes": len(script.annotations),
            }
        )

    def _write_beat(
        self,
        context: str,
        context_version: str,
        preset: StylePreset,
        outline: ScriptOutline,
        index: int,
        beat: OutlineBeat,
        written: list[ScriptSegment],
        scene_ids: list[str],
        ctx: StageContext,
    ) -> list[SegmentReply]:
        cfg = self.settings
        target = max(1, target_chars(beat.target_s, cfg.chars_per_s))
        lo, hi = round(target * (1 - BEAT_TOLERANCE)), round(target * (1 + BEAT_TOLERANCE))
        purpose = next((b.purpose for b in preset.structure if b.name == beat.beat), None)
        rules = Rules(
            scene_ids=scene_ids,
            target_chars=target,
            tolerance=BEAT_TOLERANCE,
            min_segment_chars=min(MIN_SEGMENT_CHARS, lo),  # a short beat may be one short segment
            banned_words=preset.banned_words,
            names=cfg.names,
            whole="本节",
        )
        instructions = render(
            "script_beat",
            index=index,
            count=len(outline.beats),
            beat=beat.beat,
            focus=beat.focus,
            purpose=purpose,
            scene_refs=beat.scene_refs,
            is_first=index == 1,
            hook_types=preset.hook_types,
            is_last=index == len(outline.beats),
            previous=[s.text for s in written[-PREVIOUS_SEGMENTS:]],
            target_chars=target,
            min_chars=lo,
            max_chars=hi,
            approx_segments=max(1, round(target / SEGMENT_CHARS_HINT)),
            min_seg=rules.min_segment_chars,
            max_seg=rules.max_segment_chars,
        )
        # the context comes first and is identical for every beat: that is the cacheable prefix
        messages = [Message("user", f"{context}\n\n{instructions.text}")]
        version = f"{context_version}+{instructions.version}"
        problems: list[str] = []
        for round_ in range(REPAIR_ROUNDS + 1):
            reply = self.llm.generate(
                WRITE_TASK, messages, BeatReply, prompt_version=version, max_tokens=4096
            )
            problems = [
                v.message
                for v in check([SegmentDraft(s.text, s.scene_refs) for s in reply.segments], rules)
            ]
            if not problems:
                return reply.segments
            ctx.progress(
                0.9 * (index - 1) / len(outline.beats),
                f"repairing {beat.beat} (round {round_ + 1}, {len(problems)} problems)",
            )
            messages = [
                *messages,
                Message("assistant", reply.model_dump_json()),
                Message(
                    "user",
                    "上一稿有以下问题：\n- "
                    + "\n- ".join(problems)
                    + f"\n场景编号只能用：{', '.join(scene_ids)}。"
                    + "请修改后重新输出这一节的完整 JSON。",
                ),
            ]
        raise ScriptError(
            f"beat {beat.beat!r} still breaks the rules after repair: " + "; ".join(problems)
        )
