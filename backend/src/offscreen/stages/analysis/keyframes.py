"""keyframes: shots + proxy -> shots.json with 3 frames per shot, quality metrics, colour
signatures and sprite sheets.

Per shot, frames at 10 % / 50 % / 90 % of its duration are written to `kf/<shot_id>_a|b|c.jpg`.
Each is measured (sharpness: Laplacian variance squashed to 0..1; brightness: mean luma) and the
shot's `quality` is the mean over its frames, so later stages can avoid blurry and black footage.
Each frame's colour histogram goes to `signatures.json`, so later stages (scene segmentation) can
measure how much the picture changes between shots without decoding images again. The middle
frames are also packed into sprite sheets (`sprites/sheet_NNN.jpg`, indexed by `sprites.json`)
for the shot strip of the analysis page. Paths are relative to this artifact's directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from offscreen.algo.frames import (
    HIST_BINS,
    KEYFRAME_POSITIONS,
    color_histogram,
    frame_quality,
    frame_times,
    shot_quality,
    sprite_slot,
)
from offscreen.domain.index import (
    Shot,
    Shots,
    ShotSignature,
    SpriteSheets,
    SpriteTile,
    VisualSignatures,
)
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.media.ffmpeg import FFmpegCanceled
from offscreen.media.transcode import (
    GRAY_WIDTH,
    RGB_SIZE,
    THUMB_HEIGHT,
    extract_frame,
    make_sprite_sheets,
)
from offscreen.stages.analysis.proxy import PROXY_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.store.files import write_model

KEYFRAME_DIR = "kf"
SPRITE_DIR = "sprites"
SPRITES_FILE = "sprites.json"
SIGNATURES_FILE = "signatures.json"
SUFFIXES = "abc"
SPRITE_TILE = (160, 90)
SPRITE_GRID = (10, 10)
"""Tiles per sheet: columns x rows."""
FRAMES_SHARE = 0.95
"""Share of the progress bar taken by frame extraction (the rest is the sprite sheets)."""


class KeyframesStage(Stage):
    name = "analysis.keyframes"
    version = 3
    lane: Lane = "cpu"

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.proxy", scope), ArtifactRef("analysis.shots", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {
            "asset_id": scope["asset_id"],
            "height": THUMB_HEIGHT,
            "positions": list(KEYFRAME_POSITIONS),
            "gray_width": GRAY_WIDTH,
            "rgb_size": list(RGB_SIZE),
            "hist_bins": HIST_BINS,
            "sprite_tile": list(SPRITE_TILE),
            "sprite_grid": list(SPRITE_GRID),
        }

    def run(self, ctx: StageContext) -> StageOutput:
        proxy = ctx.input("analysis.proxy").path(PROXY_FILE)
        doc = ctx.input("analysis.shots").read_model(SHOTS_FILE, Shots)
        (ctx.out_dir / KEYFRAME_DIR).mkdir()
        (ctx.out_dir / SPRITE_DIR).mkdir()
        gray_file = ctx.out_dir / ".gray"
        rgb_file = ctx.out_dir / ".rgb"
        signatures: list[ShotSignature] = []

        shots: list[Shot] = []
        try:
            for i, shot in enumerate(doc.shots):
                paths: list[str] = []
                measured: list[tuple[float, float]] = []
                hists: list[list[int]] = []
                for suffix, at_ms in zip(
                    SUFFIXES, frame_times(shot.start_ms, shot.end_ms), strict=True
                ):
                    rel = f"{KEYFRAME_DIR}/{shot.id}_{suffix}.jpg"
                    extract_frame(
                        proxy,
                        ctx.out_dir / rel,
                        at_ms=at_ms,
                        gray_to=gray_file,
                        rgb_to=rgb_file,
                        should_cancel=ctx.is_canceled,
                    )
                    measured.append(frame_quality(_read_gray(gray_file)))
                    hists.append(color_histogram(_read_rgb(rgb_file)))
                    paths.append(rel)
                signatures.append(ShotSignature(shot_id=shot.id, frames=hists))
                shots.append(
                    shot.model_copy(update={"keyframes": paths, "quality": shot_quality(measured)})
                )
                ctx.progress(FRAMES_SHARE * (i + 1) / len(doc.shots), shot.id)
            gray_file.unlink(missing_ok=True)
            rgb_file.unlink(missing_ok=True)

            sprites = self._sprite_sheets(ctx, doc.asset_id, shots)
        except FFmpegCanceled as e:
            raise StageCanceled(self.name) from e

        write_model(ctx.out_dir / SHOTS_FILE, doc.model_copy(update={"shots": shots}))
        write_model(ctx.out_dir / SPRITES_FILE, sprites)
        write_model(
            ctx.out_dir / SIGNATURES_FILE,
            VisualSignatures(
                asset_id=doc.asset_id, bins_per_channel=HIST_BINS, signatures=signatures
            ),
        )
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={
                "shots": len(shots),
                "keyframes": 3 * len(shots),
                "sprite_sheets": len(sprites.sheets),
            }
        )

    def _sprite_sheets(self, ctx: StageContext, asset_id: str, shots: list[Shot]) -> SpriteSheets:
        columns, rows = SPRITE_GRID
        tile_w, tile_h = SPRITE_TILE
        middle = [ctx.out_dir / s.keyframes[1] for s in shots]  # the 50 % frame of each shot
        make_sprite_sheets(
            middle,
            ctx.out_dir / SPRITE_DIR / "sheet_%03d.jpg",
            tile_width=tile_w,
            tile_height=tile_h,
            columns=columns,
            rows=rows,
            should_cancel=ctx.is_canceled,
        )
        sheets = sorted(p.name for p in (ctx.out_dir / SPRITE_DIR).glob("sheet_*.jpg"))
        tiles = [
            SpriteTile(shot_id=s.id, sheet=sheet, col=col, row=row)
            for i, s in enumerate(shots)
            for sheet, col, row in [sprite_slot(i, columns, rows)]
        ]
        return SpriteSheets(
            asset_id=asset_id,
            tile_width=tile_w,
            tile_height=tile_h,
            columns=columns,
            rows=rows,
            sheets=[f"{SPRITE_DIR}/{name}" for name in sheets],
            tiles=tiles,
        )


def _read_gray(path: Path) -> NDArray[np.uint8]:
    """The raw grayscale frame ffmpeg wrote, as a (height, GRAY_WIDTH) array."""
    data = np.fromfile(path, dtype=np.uint8)
    return data.reshape(-1, GRAY_WIDTH)


def _read_rgb(path: Path) -> NDArray[np.uint8]:
    """The raw rgb24 frame ffmpeg wrote, as a (height, width, 3) array."""
    w, h = RGB_SIZE
    return np.fromfile(path, dtype=np.uint8).reshape(h, w, 3)
