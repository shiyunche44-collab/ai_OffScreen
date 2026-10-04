"""story: transcript (+ shots) -> story.json and a coarse scenes.json (v0).

Dialogue is cut into chunks by time and size, the LLM summarizes each chunk (task
`story_chunk`), and the summaries are merged into one story (task `story`). Every chunk is
one coarse scene whose borders are snapped to shot starts, so M1's shot picking has scene
and shot ids to refer to. The real scene segmentation (M3-11) will replace this part and
make this stage read scenes.json instead of writing it. Every plot claim in the story
cites scene ids, and the stage checks that they exist."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, field_validator

from offscreen.algo.story import ScenePlan, check_story_refs, fmt_clock, plan_scenes
from offscreen.domain.index import (
    Act,
    Scene,
    Scenes,
    Shot,
    Shots,
    Story,
    Transcript,
    TranscriptLine,
    TurningPoint,
)
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.prompts import render, template_version
from offscreen.providers.ports import LLM, Message
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.store.files import write_model

STORY_FILE = "story.json"
SCENES_FILE = "scenes.json"
CHUNK_TASK = "story_chunk"
STORY_TASK = "story"
MAX_CHUNK_CHARS = 6_000
MAX_CHUNK_SPAN_MS = 15 * 60 * 1000
MERGE_ATTEMPTS = 2


class StoryError(RuntimeError):
    pass


class ChunkReply(BaseModel):
    """What the model returns for one chunk (lenient: unknown fields are ignored)."""

    summary: str = Field(min_length=1)
    characters: list[str] = []
    location: str | None = None
    importance: float = 0.5

    @field_validator("importance")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return min(1.0, max(0.0, v))

    @field_validator("location")
    @classmethod
    def _blank_location(cls, v: str | None) -> str | None:
        return v.strip() or None if v else None


class ActReply(BaseModel):
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    scene_ids: list[str] = Field(min_length=1)


class TurningPointReply(BaseModel):
    scene_id: str
    what: str = Field(min_length=1)


class StoryReply(BaseModel):
    logline: str = Field(min_length=1)
    synopsis: str = Field(min_length=1)
    acts: list[ActReply] = Field(min_length=1)
    turning_points: list[TurningPointReply] = []
    ending: str | None = None
    themes: list[str] = []


class StoryStage(Stage):
    name = "analysis.story"
    version = 1
    lane: Lane = "api"

    def __init__(self, llm: LLM, models: Mapping[str, str] | None = None) -> None:
        self.llm = llm
        self.models = dict(models or {})
        """`task -> "provider/model"` for the tasks this stage uses; part of the cache key."""

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.transcript", scope), ArtifactRef("analysis.shots", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {
            "asset_id": scope["asset_id"],
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "max_chunk_span_ms": MAX_CHUNK_SPAN_MS,
            "language": "zh",
        }

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "models": {t: self.models.get(t) for t in (CHUNK_TASK, STORY_TASK)},
            "prompts": {
                CHUNK_TASK: template_version("story_chunk"),
                STORY_TASK: template_version("story_merge"),
            },
        }

    def run(self, ctx: StageContext) -> StageOutput:
        asset_id = ctx.scope["asset_id"]
        transcript = ctx.input("analysis.transcript").read_model(TRANSCRIPT_FILE, Transcript)
        shots = ctx.input("analysis.shots").read_model(SHOTS_FILE, Shots).shots
        lines = sorted(transcript.lines, key=lambda x: (x.start_ms, x.end_ms))
        if not lines:
            raise StoryError("no dialogue in the transcript: nothing to build a story from")
        if not shots:
            raise StoryError("no shots detected")

        plans = plan_scenes(
            lines,
            [s.start_ms for s in shots],
            max_chars=MAX_CHUNK_CHARS,
            max_span_ms=MAX_CHUNK_SPAN_MS,
        )
        width = max(3, len(str(len(plans))))
        scenes: list[Scene] = []
        replies: list[ChunkReply] = []
        for i, plan in enumerate(plans):
            if ctx.is_canceled():
                raise StageCanceled(self.name)
            scene_lines = lines[plan.line_lo : plan.line_hi]
            start_ms, end_ms = shots[plan.shot_lo].start_ms, shots[plan.shot_hi - 1].end_ms
            reply = self._summarize_chunk(
                i, len(plans), start_ms, end_ms, scene_lines, replies[-1] if replies else None
            )
            replies.append(reply)
            scenes.append(self._scene(f"sc_{i + 1:0{width}d}", plan, reply, lines, shots))
            ctx.progress((i + 1) / (len(plans) + 1) * 0.95, f"summarized chunk {i + 1}")

        story = self._merge(asset_id, scenes, replies)
        write_model(ctx.out_dir / SCENES_FILE, Scenes(asset_id=asset_id, scenes=scenes))
        write_model(ctx.out_dir / STORY_FILE, story)
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={"lines": len(lines), "scenes": len(scenes), "acts": len(story.acts)}
        )

    # ---- per chunk ---------------------------------------------------------------------
    def _summarize_chunk(
        self,
        index: int,
        total: int,
        start_ms: int,
        end_ms: int,
        lines: list[TranscriptLine],
        previous: ChunkReply | None,
    ) -> ChunkReply:
        prompt = render(
            "story_chunk",
            index=index + 1,
            total=total,
            start=fmt_clock(start_ms),
            end=fmt_clock(end_ms),
            previous_summary=previous.summary if previous else None,
            lines=[{"at": fmt_clock(x.start_ms), "text": x.text} for x in lines],
        )
        return self.llm.generate(
            CHUNK_TASK,
            [Message("user", prompt.text)],
            ChunkReply,
            prompt_version=prompt.version,
            max_tokens=1024,
        )

    @staticmethod
    def _scene(
        scene_id: str,
        plan: ScenePlan,
        reply: ChunkReply,
        lines: list[TranscriptLine],
        shots: list[Shot],
    ) -> Scene:
        return Scene(
            id=scene_id,
            start_ms=shots[plan.shot_lo].start_ms,
            end_ms=shots[plan.shot_hi - 1].end_ms,
            shot_ids=[s.id for s in shots[plan.shot_lo : plan.shot_hi]],
            line_ids=[x.id for x in lines[plan.line_lo : plan.line_hi]],
            summary=reply.summary,
            location=reply.location,
            importance=reply.importance,
        )

    # ---- merge -------------------------------------------------------------------------
    def _merge(self, asset_id: str, scenes: list[Scene], replies: list[ChunkReply]) -> Story:
        prompt = render(
            "story_merge",
            scenes=[
                {
                    "id": s.id,
                    "start": fmt_clock(s.start_ms),
                    "end": fmt_clock(s.end_ms),
                    "location": s.location,
                    "importance": s.importance,
                    "characters": r.characters,
                    "summary": s.summary,
                }
                for s, r in zip(scenes, replies, strict=True)
            ],
        )
        messages = [Message("user", prompt.text)]
        scene_ids = [s.id for s in scenes]
        problems: list[str] = []
        for _ in range(MERGE_ATTEMPTS):
            reply = self.llm.generate(
                STORY_TASK, messages, StoryReply, prompt_version=prompt.version, max_tokens=4096
            )
            problems = check_story_refs(
                [a.scene_ids for a in reply.acts],
                [t.scene_id for t in reply.turning_points],
                scene_ids,
            )
            if not problems:
                return Story(
                    asset_id=asset_id,
                    logline=reply.logline,
                    synopsis=reply.synopsis,
                    acts=[
                        Act(name=a.name, scene_ids=a.scene_ids, summary=a.summary)
                        for a in reply.acts
                    ],
                    turning_points=[
                        TurningPoint(scene_id=t.scene_id, what=t.what) for t in reply.turning_points
                    ],
                    ending=reply.ending,
                    themes=reply.themes,
                )
            messages = [
                *messages,
                Message("assistant", reply.model_dump_json()),
                Message(
                    "user",
                    "上一次输出有以下问题：\n- "
                    + "\n- ".join(problems)
                    + f"\n请只使用这些场景编号：{', '.join(scene_ids)}。重新输出完整 JSON。",
                ),
            ]
        raise StoryError("story cites invalid scene ids after retry: " + "; ".join(problems))
