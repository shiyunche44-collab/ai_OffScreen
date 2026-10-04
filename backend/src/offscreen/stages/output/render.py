"""render: timeline -> final.mp4 (T5, deterministic: no model is called).

Reads only the compile artifact (timeline + the narration files beside it) and the source
film the timeline's items point to. Steps (ARCHITECTURE §7.5), all in a scratch directory
that is removed afterwards:
1. each video item becomes a silent clip of exactly `f1 - f0` frames in one common encoding;
2. the clips are joined without re-encoding;
3. all sound (narration, and the film's own audio under it) is mixed as a whole into one wav;
4. the video is re-encoded once with the subtitles burned in, with the mixdown as AAC.

v0 keeps the source frame size (layout "keep"), takes the film's first audio stream, and
does no loudness normalization (M6)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from offscreen.algo.ass import FONT, build_ass
from offscreen.domain.common import Rational
from offscreen.domain.job import Lane
from offscreen.domain.timeline import Timeline
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.media import render as media
from offscreen.media.ffmpeg import FFmpegCanceled
from offscreen.stages.output.compile import TIMELINE_FILE
from offscreen.store.repos import AssetRepo

FINAL_FILE = "final.mp4"
SUBTITLES_FILE = "subtitles.ass"


class RenderError(RuntimeError):
    pass


def frames_to_ms(frames: int, fps: Rational) -> int:
    return round(frames * 1000 * fps.den / fps.num)


class RenderStage(Stage):
    name = "output.render"
    version = 1
    lane: Lane = "cpu"

    def __init__(self, assets: AssetRepo, font: str = FONT) -> None:
        self.assets = assets
        self.font = font

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("output.compile", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        asset = self.assets.get(scope["asset_id"])
        if asset is None:
            raise RenderError(f"unknown asset {scope['asset_id']}")
        return {
            "asset_id": scope["asset_id"],
            # The film is read from outside the artifact graph, so its identity is in the key.
            "source_fingerprint": asset.fingerprint,
            "font": self.font,
            "clip_crf": media.CLIP_CRF,
            "final_crf": media.FINAL_CRF,
            "preset": media.PRESET,
            "audio_bitrate": media.AUDIO_BITRATE,
            "mix_rate": media.MIX_RATE,
            "fade_ms": media.FADE_MS,
        }

    def run(self, ctx: StageContext) -> StageOutput:
        asset_id = ctx.scope["asset_id"]
        asset = self.assets.get(asset_id)
        if asset is None:
            raise RenderError(f"unknown asset {asset_id}")
        source = Path(asset.source_path)
        if not source.is_file():
            raise RenderError(f"source film not found: {source} (moved? re-import it)")

        compiled = ctx.input("output.compile")
        tl = compiled.read_model(TIMELINE_FILE, Timeline)
        out = tl.output
        if out.layout != "keep":
            raise RenderError(f"layout {out.layout!r} is not supported yet")
        if any(v.asset_id != asset_id for v in tl.video):
            raise RenderError("timeline uses footage from another asset")
        fps = out.fps
        total_ms = frames_to_ms(tl.duration_frames, fps)

        try:
            with tempfile.TemporaryDirectory(dir=ctx.out_dir, prefix=".work-") as tmp:
                work = Path(tmp)
                self._render(ctx, tl, compiled.dir, source, bool(asset.audio), work, total_ms)
        except FFmpegCanceled as e:
            raise StageCanceled(self.name) from e
        final = ctx.out_dir / FINAL_FILE
        return StageOutput(
            meta={
                "duration_frames": tl.duration_frames,
                "duration_ms": total_ms,
                "bytes": final.stat().st_size,
                "subtitles": len(tl.subtitles),
            }
        )

    def _render(
        self,
        ctx: StageContext,
        tl: Timeline,
        compiled_dir: Path,
        source: Path,
        has_audio: bool,
        work: Path,
        total_ms: int,
    ) -> None:
        fps, out = tl.output.fps, tl.output
        cancel = ctx.is_canceled

        # 1. clips
        clip_files: list[Path] = []
        for i, v in enumerate(tl.video):
            dst = work / f"clip_{i:05d}.mp4"
            media.render_clip(
                source,
                dst,
                src_in_ms=v.src_in_ms,
                frames=v.f1 - v.f0,
                fps=fps,
                width=out.width,
                height=out.height,
                speed=v.speed,
                should_cancel=cancel,
            )
            clip_files.append(dst)
            ctx.progress(0.6 * (i + 1) / len(tl.video), f"clip {i + 1}/{len(tl.video)}")

        # 2. concat
        silent = work / "silent.mp4"
        media.concat_videos(clip_files, silent, list_file=work / "clips.txt", should_cancel=cancel)
        for f in clip_files:
            f.unlink()
        ctx.progress(0.65, "joined")

        # 3. mixdown
        parts = [
            media.AudioPart(
                path=compiled_dir / n.file, start_ms=frames_to_ms(n.f0, fps), gain_db=n.gain_db
            )
            for n in tl.narration
        ]
        if has_audio:
            parts += [
                media.AudioPart(
                    path=source,
                    start_ms=frames_to_ms(s.f0, fps),
                    gain_db=s.gain_db,
                    seek_ms=s.src_in_ms,
                    duration_ms=frames_to_ms(s.f1 - s.f0, fps),
                    fade=True,
                )
                for s in tl.source_audio
            ]
        if not parts:
            raise RenderError("timeline has no sound at all")
        for p in parts:
            if not p.path.is_file():
                raise RenderError(f"audio file missing: {p.path}")
        mix = work / "mix.wav"
        media.mix_audio(
            parts, mix, total_ms=total_ms, script_file=work / "mix.filter", should_cancel=cancel
        )
        ctx.progress(0.8, "mixed")

        # 4. final encode with subtitles
        subs: Path | None = None
        if tl.subtitles:
            subs = ctx.out_dir / SUBTITLES_FILE
            subs.write_text(
                build_ass(tl.subtitles, fps, out.width, out.height, self.font), encoding="utf-8"
            )
        media.encode_final(
            silent,
            mix,
            ctx.out_dir / FINAL_FILE,
            subtitles=subs,
            duration_ms=total_ms,
            on_progress=lambda f: ctx.progress(0.8 + 0.2 * f, "encoding"),
            should_cancel=cancel,
        )
