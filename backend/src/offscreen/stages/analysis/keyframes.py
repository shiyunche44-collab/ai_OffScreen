"""keyframes: shots + proxy -> shots.json with one thumbnail per shot (v0).

Writes `kf/<shot_id>.jpg` of each shot's middle frame, and the same `Shots` document with
`keyframes` filled in. Paths are relative to this artifact's directory. The 3-frame sampling
and the sharpness / brightness metrics are M3-03."""

from __future__ import annotations

from typing import Any

from offscreen.domain.index import Shot, Shots
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.media.ffmpeg import FFmpegCanceled
from offscreen.media.transcode import THUMB_HEIGHT, extract_frame
from offscreen.stages.analysis.proxy import PROXY_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.store.files import write_model

KEYFRAME_DIR = "kf"


class KeyframesStage(Stage):
    name = "analysis.keyframes"
    version = 1
    lane: Lane = "cpu"

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.proxy", scope), ArtifactRef("analysis.shots", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"asset_id": scope["asset_id"], "height": THUMB_HEIGHT, "position": "middle"}

    def run(self, ctx: StageContext) -> StageOutput:
        proxy = ctx.input("analysis.proxy").path(PROXY_FILE)
        doc = ctx.input("analysis.shots").read_model(SHOTS_FILE, Shots)
        (ctx.out_dir / KEYFRAME_DIR).mkdir()

        shots: list[Shot] = []
        try:
            for i, shot in enumerate(doc.shots):
                rel = f"{KEYFRAME_DIR}/{shot.id}.jpg"
                extract_frame(
                    proxy,
                    ctx.out_dir / rel,
                    at_ms=shot.start_ms + shot.duration_ms // 2,
                    should_cancel=ctx.is_canceled,
                )
                shots.append(shot.model_copy(update={"keyframes": [rel]}))
                ctx.progress((i + 1) / len(doc.shots), shot.id)
        except FFmpegCanceled as e:
            raise StageCanceled(self.name) from e

        write_model(ctx.out_dir / SHOTS_FILE, doc.model_copy(update={"shots": shots}))
        return StageOutput(meta={"keyframes": len(shots)})
