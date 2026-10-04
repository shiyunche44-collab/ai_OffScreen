"""shots: proxy -> shots.json (ARCHITECTURE §7.1).

The detector proposes cuts; `algo.shots.normalize_shots` makes them a partition of the
whole video, merging fragments under 0.5 s and splitting shots over 8 s."""

from __future__ import annotations

from typing import Any

from offscreen.algo.shots import MAX_SHOT_MS, MIN_SHOT_MS, normalize_shots
from offscreen.domain.index import Shot, Shots
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.media.probe import ProbeError, probe
from offscreen.providers.ports import DetectionCanceled, ShotDetector
from offscreen.stages.analysis.proxy import PROXY_FILE
from offscreen.store.files import write_model
from offscreen.store.repos import AssetRepo

SHOTS_FILE = "shots.json"


class ShotsError(RuntimeError):
    pass


class ShotsStage(Stage):
    name = "analysis.shots"
    version = 1
    lane: Lane = "cpu"

    def __init__(self, assets: AssetRepo, detector: ShotDetector) -> None:
        self.assets = assets
        self.detector = detector

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.proxy", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {
            "asset_id": scope["asset_id"],
            "detector": self.detector.id,
            "min_ms": MIN_SHOT_MS,
            "max_ms": MAX_SHOT_MS,
        }

    def run(self, ctx: StageContext) -> StageOutput:
        asset = self.assets.get(ctx.scope["asset_id"])
        if asset is None:
            raise ShotsError(f"unknown asset {ctx.scope['asset_id']}")
        proxy = ctx.input("analysis.proxy").path(PROXY_FILE)
        try:
            # The proxy may differ from the source by a few ms; shots must end where
            # the video we actually hand to the next stages ends.
            duration_ms = probe(proxy).duration_ms
        except ProbeError as e:
            raise ShotsError(str(e)) from e

        try:
            raw = self.detector.detect(
                proxy,
                on_progress=lambda f: ctx.progress(f * 0.95, "detecting cuts"),
                should_cancel=ctx.is_canceled,
            )
        except DetectionCanceled as e:
            raise StageCanceled(self.name) from e

        spans = normalize_shots(
            [(r.start_ms, r.end_ms) for r in raw],
            duration_ms,
            min_ms=MIN_SHOT_MS,
            max_ms=MAX_SHOT_MS,
        )
        width = max(4, len(str(len(spans))))
        shots = Shots(
            asset_id=asset.id,
            shots=[
                Shot(id=f"sh_{i:0{width}d}", start_ms=a, end_ms=b)
                for i, (a, b) in enumerate(spans, 1)
            ],
        )
        write_model(ctx.out_dir / SHOTS_FILE, shots)
        ctx.progress(1.0, "done")
        return StageOutput(meta={"shots": len(spans), "raw_cuts": max(0, len(raw) - 1)})
