"""compile: plan -> timeline.json (T4, deterministic: no model is called).

The output keeps the source film's frame size and frame rate (M1: profile "source"); the
asset record supplies them, and the resolved spec is part of the cache key. Subtitle lines
are limited to 22 characters on landscape output and 14 on portrait (ARCHITECTURE §7.4)."""

from __future__ import annotations

from typing import Any

from offscreen.algo.compile import compile_timeline
from offscreen.domain.job import Lane
from offscreen.domain.plan import EditPlan
from offscreen.domain.timeline import OutputSpec
from offscreen.engine import ArtifactRef, Scope, Stage, StageContext, StageOutput
from offscreen.stages.creation.plan import PLAN_FILE
from offscreen.store.files import write_model
from offscreen.store.repos import AssetRepo

TIMELINE_FILE = "timeline.json"
SUBTITLE_CHARS_LANDSCAPE = 22
SUBTITLE_CHARS_PORTRAIT = 14


class CompileStageError(RuntimeError):
    pass


class CompileStage(Stage):
    name = "output.compile"
    version = 1
    lane: Lane = "cpu"

    def __init__(self, assets: AssetRepo) -> None:
        self.assets = assets

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("creation.plan", scope)]

    def _spec(self, scope: Scope) -> OutputSpec:
        asset = self.assets.get(scope["asset_id"])
        if asset is None:
            raise CompileStageError(f"unknown asset {scope['asset_id']}")
        v = asset.video
        return OutputSpec(
            profile="source", width=v.width, height=v.height, fps=v.fps, layout="keep"
        )

    def params(self, scope: Scope) -> dict[str, Any]:
        spec = self._spec(scope)
        return {
            "asset_id": scope["asset_id"],
            "output": spec.model_dump(mode="json"),
            "subtitle_chars": [SUBTITLE_CHARS_LANDSCAPE, SUBTITLE_CHARS_PORTRAIT],
        }

    def run(self, ctx: StageContext) -> StageOutput:
        plan = ctx.input("creation.plan").read_model(PLAN_FILE, EditPlan)
        spec = self._spec(ctx.scope)
        limit = SUBTITLE_CHARS_LANDSCAPE if spec.width >= spec.height else SUBTITLE_CHARS_PORTRAIT
        timeline = compile_timeline(plan, spec, subtitle_max_chars=limit)
        write_model(ctx.out_dir / TIMELINE_FILE, timeline)
        return StageOutput(
            meta={
                "duration_frames": timeline.duration_frames,
                "video_items": len(timeline.video),
                "subtitles": len(timeline.subtitles),
            }
        )
