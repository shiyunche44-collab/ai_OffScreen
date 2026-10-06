"""review: script + story + scenes -> review.json (fact-check annotations).

One LLM call (task `script_critic`, by default another provider than the writer so the two do
not share blind spots) compares the script with the story and the scenes and reports the
statements the material does not support. It only annotates: the text is never changed. Findings
that name a segment the script does not have are dropped."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from offscreen.algo.script import key_lines, scene_lines
from offscreen.algo.story import fmt_clock
from offscreen.domain.index import Scenes, Story, Transcript
from offscreen.domain.job import Lane
from offscreen.domain.script import Annotation, Script, ScriptReview
from offscreen.engine import ArtifactRef, Scope, Stage, StageContext, StageOutput
from offscreen.prompts import render, template_version
from offscreen.providers.ports import LLM, Message
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.story import STORY_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.stages.creation.script import SCRIPT_FILE
from offscreen.store.files import write_model

REVIEW_FILE = "review.json"
REVIEW_TASK = "script_critic"
MAX_MESSAGE_CHARS = 300


class Finding(BaseModel):
    """One reported problem (lenient: unknown fields are ignored)."""

    segment_id: str
    problem: str = Field(min_length=1)


class ReviewReply(BaseModel):
    findings: list[Finding] = []


class ReviewStage(Stage):
    name = "creation.review"
    version = 1
    lane: Lane = "api"

    def __init__(self, llm: LLM, models: Mapping[str, str] | None = None) -> None:
        self.llm = llm
        self.models = dict(models or {})
        """`task -> "provider/model"` for the tasks this stage uses; part of the cache key."""

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [
            ArtifactRef("creation.script", scope),
            ArtifactRef("analysis.story", scope),
            ArtifactRef("analysis.scenes", scope),
            ArtifactRef("analysis.transcript", scope),
        ]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"asset_id": scope["asset_id"], "max_message_chars": MAX_MESSAGE_CHARS}

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "models": {REVIEW_TASK: self.models.get(REVIEW_TASK)},
            "prompts": {REVIEW_TASK: template_version("script_review")},
        }

    def run(self, ctx: StageContext) -> StageOutput:
        script = ctx.input("creation.script").read_model(SCRIPT_FILE, Script)
        story = ctx.input("analysis.story").read_model(STORY_FILE, Story)
        scenes = ctx.input("analysis.scenes").read_model(SCENES_FILE, Scenes).scenes
        lines = ctx.input("analysis.transcript").read_model(TRANSCRIPT_FILE, Transcript).lines

        prompt = render(
            "script_review",
            spoil_ending=script.params.spoil_ending,
            logline=story.logline,
            synopsis=story.synopsis,
            turning_points=story.turning_points,
            ending=story.ending,
            scenes=[
                {
                    "id": s.id,
                    "start": fmt_clock(s.start_ms),
                    "end": fmt_clock(s.end_ms),
                    "summary": s.summary,
                    "lines": key_lines(scene_lines(s, lines)),
                }
                for s in scenes
            ],
            segments=[
                {"id": g.id, "scene_refs": g.scene_refs, "text": g.text}
                for g in script.segments
                if g.kind == "narration"
            ],
        )
        ctx.progress(0.1, "reviewing")
        reply = self.llm.generate(
            REVIEW_TASK,
            [Message("user", prompt.text)],
            ReviewReply,
            prompt_version=prompt.version,
            max_tokens=4096,
        )

        known = {g.id for g in script.segments}
        kept = [f for f in reply.findings if f.segment_id in known]
        review = ScriptReview(
            asset_id=ctx.scope["asset_id"],
            annotations=[
                Annotation(
                    segment_id=f.segment_id,
                    type="fact_check",
                    message=_cap(f.problem.strip()),
                )
                for f in kept
            ],
        )
        write_model(ctx.out_dir / REVIEW_FILE, review)
        ctx.progress(1.0, "done")
        return StageOutput(meta={"findings": len(kept), "dropped": len(reply.findings) - len(kept)})


def _cap(text: str) -> str:
    return text if len(text) <= MAX_MESSAGE_CHARS else text[: MAX_MESSAGE_CHARS - 1] + "…"
