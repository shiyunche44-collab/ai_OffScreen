"""story: scenes -> story.json, a layered summary of the film (M3-12).

Two levels over the scene summaries of `analysis.scenes`: first the scenes are grouped into acts
and the key turning points are named (prompt `story_acts`); then the acts become the film's
logline, synopsis, ending and themes (prompt `story_synthesis`; both use task `story`). Every
claim about the plot is anchored in scene ids and the stage checks them: acts must contain every
scene exactly once, turning points and the ending must cite scenes that exist. A wrong answer
gets one repair round with the problems listed, then the stage fails rather than keep a story
that points at nothing.

Relations and character biographies need characters (M3-08 / M3-09) and stay empty until then."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from offscreen.algo.story import check_story_refs, fmt_clock
from offscreen.domain.index import Act, Scene, Scenes, Story, TurningPoint
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.prompts import render, template_version
from offscreen.providers.ports import LLM, Message
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.store.files import write_model

STORY_FILE = "story.json"
STORY_TASK = "story"
ATTEMPTS = 2

R = TypeVar("R", bound=BaseModel)


class StoryError(RuntimeError):
    pass


class ActReply(BaseModel):
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    scene_ids: list[str] = Field(min_length=1)


class TurningPointReply(BaseModel):
    scene_id: str
    what: str = Field(min_length=1)


class ActsReply(BaseModel):
    acts: list[ActReply] = Field(min_length=1)
    turning_points: list[TurningPointReply] = []


class SynthesisReply(BaseModel):
    logline: str = Field(min_length=1)
    synopsis: str = Field(min_length=1)
    ending: str | None = None
    ending_scene_ids: list[str] = []
    themes: list[str] = []


class StoryStage(Stage):
    name = "analysis.story"
    version = 2
    lane: Lane = "api"

    def __init__(self, llm: LLM, models: Mapping[str, str] | None = None) -> None:
        self.llm = llm
        self.models = dict(models or {})
        """`task -> "provider/model"` for the tasks this stage uses; part of the cache key."""

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.scenes", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"asset_id": scope["asset_id"], "language": "zh"}

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "models": {STORY_TASK: self.models.get(STORY_TASK)},
            "prompts": {
                "acts": template_version("story_acts"),
                "synthesis": template_version("story_synthesis"),
            },
        }

    def run(self, ctx: StageContext) -> StageOutput:
        doc = ctx.input("analysis.scenes").read_model(SCENES_FILE, Scenes)
        scenes = doc.scenes
        if not scenes:
            raise StoryError("no scenes to build a story from")
        ids = [s.id for s in scenes]

        acts = self._acts(scenes, ids)
        ctx.progress(0.5, "acts and turning points")
        if ctx.is_canceled():
            raise StageCanceled(self.name)
        synthesis = self._synthesis(scenes, ids, acts)

        story = Story(
            asset_id=doc.asset_id,
            logline=synthesis.logline,
            synopsis=synthesis.synopsis,
            acts=[Act(name=a.name, scene_ids=a.scene_ids, summary=a.summary) for a in acts.acts],
            turning_points=[
                TurningPoint(scene_id=t.scene_id, what=t.what) for t in acts.turning_points
            ],
            ending=synthesis.ending,
            ending_scene_ids=synthesis.ending_scene_ids,
            themes=synthesis.themes,
        )
        write_model(ctx.out_dir / STORY_FILE, story)
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={
                "scenes": len(scenes),
                "acts": len(story.acts),
                "turning_points": len(story.turning_points),
            }
        )

    # ---- level 1: scenes -> acts ---------------------------------------------------------
    def _acts(self, scenes: Sequence[Scene], ids: list[str]) -> ActsReply:
        prompt = render(
            "story_acts",
            scenes=[
                {
                    "id": s.id,
                    "start": fmt_clock(s.start_ms),
                    "end": fmt_clock(s.end_ms),
                    "location": s.location,
                    "importance": s.importance,
                    "summary": s.summary,
                }
                for s in scenes
            ],
        )

        def problems(r: ActsReply) -> list[str]:
            return check_story_refs(
                [a.scene_ids for a in r.acts],
                [t.scene_id for t in r.turning_points],
                ids,
                cover=True,
            )

        return self._ask(prompt.text, prompt.version, ActsReply, problems, ids, 4096)

    # ---- level 2: acts -> the whole story ------------------------------------------------
    def _synthesis(
        self, scenes: Sequence[Scene], ids: list[str], acts: ActsReply
    ) -> SynthesisReply:
        by_id = {s.id: s for s in scenes}
        prompt = render(
            "story_synthesis",
            acts=[
                {
                    "name": a.name,
                    "summary": a.summary,
                    "scene_ids": a.scene_ids,
                    "start": fmt_clock(by_id[a.scene_ids[0]].start_ms),
                    "end": fmt_clock(by_id[a.scene_ids[-1]].end_ms),
                }
                for a in acts.acts
            ],
            turning_points=acts.turning_points,
        )

        def problems(r: SynthesisReply) -> list[str]:
            found = check_story_refs([], [], ids, ending_ids=r.ending_scene_ids)
            if r.ending and not r.ending_scene_ids:
                found.append("写了结局，但没有给出 ending_scene_ids（结局依据的场景编号）")
            return found

        return self._ask(prompt.text, prompt.version, SynthesisReply, problems, ids, 4096)

    # ---- ask, check, repair once ---------------------------------------------------------
    def _ask(
        self,
        text: str,
        prompt_version: str,
        schema: type[R],
        problems: Callable[[R], list[str]],
        scene_ids: Sequence[str],
        max_tokens: int,
    ) -> R:
        messages = [Message("user", text)]
        found: list[str] = []
        for _ in range(ATTEMPTS):
            reply = self.llm.generate(
                STORY_TASK, messages, schema, prompt_version=prompt_version, max_tokens=max_tokens
            )
            found = problems(reply)
            if not found:
                return reply
            messages = [
                *messages,
                Message("assistant", reply.model_dump_json()),
                Message(
                    "user",
                    "上一次输出有以下问题：\n- "
                    + "\n- ".join(found)
                    + f"\n请只使用这些场景编号：{', '.join(scene_ids)}。重新输出完整 JSON。",
                ),
            ]
        raise StoryError("story cites invalid scene ids after retry: " + "; ".join(found))
