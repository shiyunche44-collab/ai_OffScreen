"""render: timeline -> final.mp4 (T5, deterministic: no model is called).

Reads only the compile artifact (timeline + the narration files beside it) and the source
film the timeline's items point to. Steps (ARCHITECTURE §7.5), all in a scratch directory
that is removed afterwards:
1. each video item becomes a silent clip of exactly `f1 - f0` frames in one common encoding;
2. the clips are joined without re-encoding;
3. all sound (narration, the film's own audio under it, music ducked under the narration) is
   mixed as a whole, sample by sample, into one wav (`algo/mixing.py`);
4. the video is re-encoded once with the subtitles burned in, with the mixdown as AAC.

v0 keeps the source frame size (layout "keep"), takes the film's first audio stream, and does
no loudness normalization yet (M6-07). The timeline's music is only rendered when a music folder
is given; nothing compiles music into a timeline before M6-08."""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from offscreen.algo.ass import FONT, build_ass
from offscreen.algo.mixing import (
    Audio,
    Layer,
    MixReport,
    MixSettings,
    Placement,
    frame_to_sample,
)
from offscreen.algo.mixing import mix as mixdown
from offscreen.domain.common import Rational
from offscreen.domain.job import Lane
from offscreen.domain.timeline import Timeline
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.media import audio as audio_io
from offscreen.media import render as media
from offscreen.media.ffmpeg import FFmpegCanceled
from offscreen.stages.output.compile import TIMELINE_FILE
from offscreen.store.repos import AssetRepo

FINAL_FILE = "final.mp4"
DEFAULT_MIX = MixSettings()
SUBTITLES_FILE = "subtitles.ass"


class RenderError(RuntimeError):
    pass


def frames_to_ms(frames: int, fps: Rational) -> int:
    return round(frames * 1000 * fps.den / fps.num)


class RenderStage(Stage):
    name = "output.render"
    version = 2  # 2: numpy mixdown (edge fades, music ducked under the narration)
    lane: Lane = "cpu"

    def __init__(self, assets: AssetRepo, font: str = FONT, mix: MixSettings = DEFAULT_MIX) -> None:
        self.assets = assets
        self.font = font
        self.mix = mix

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
            "mix": dataclasses.asdict(self.mix),
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
        total_ms = frames_to_ms(tl.duration_frames, out.fps)

        try:
            with tempfile.TemporaryDirectory(dir=ctx.out_dir, prefix=".work-") as tmp:
                work = Path(tmp)
                report = render_timeline(
                    tl,
                    compiled.dir,
                    ctx.out_dir,
                    work,
                    video=source,
                    audio=source if asset.audio else None,
                    font=self.font,
                    mix=self.mix,
                    progress=ctx.progress,
                    should_cancel=ctx.is_canceled,
                )
        except FFmpegCanceled as e:
            raise StageCanceled(self.name) from e
        final = ctx.out_dir / FINAL_FILE
        return StageOutput(
            meta={
                "duration_frames": tl.duration_frames,
                "duration_ms": total_ms,
                "bytes": final.stat().st_size,
                "subtitles": len(tl.subtitles),
                "mix_peak": round(report.peak, 4),
                "mix_clipped_samples": report.clipped,
            }
        )


@dataclass(frozen=True)
class Quality:
    """Encoder settings: the final video's, or a preview's (fast and small)."""

    preset: str = media.PRESET
    clip_crf: int = media.CLIP_CRF
    final_crf: int = media.FINAL_CRF


DEFAULT_QUALITY = Quality()


def render_timeline(
    tl: Timeline,
    narration_dir: Path,
    out_dir: Path,
    work: Path,
    *,
    video: Path,
    audio: Path | None,
    font: str,
    quality: Quality = DEFAULT_QUALITY,
    mix: MixSettings = DEFAULT_MIX,
    bgm_dir: Path | None = None,
    progress: Callable[[float, str], None] = lambda _f, _m: None,
    should_cancel: Callable[[], bool] | None = None,
) -> MixReport:
    """Timeline -> `out_dir/final.mp4` (and the subtitle file), working in `work`.

    `video` supplies the picture, `audio` the film's own sound (None: the film has none); the
    narration files are found next to `narration_dir`, the music in `bgm_dir`. Used for the final
    render (the source film) and for single-segment previews (the proxy)."""
    fps, out = tl.output.fps, tl.output
    total_ms = frames_to_ms(tl.duration_frames, fps)
    cancel = should_cancel

    # 1. clips
    clip_files: list[Path] = []
    for i, v in enumerate(tl.video):
        dst = work / f"clip_{i:05d}.mp4"
        media.render_clip(
            video,
            dst,
            src_in_ms=v.src_in_ms,
            frames=v.f1 - v.f0,
            fps=fps,
            width=out.width,
            height=out.height,
            speed=v.speed,
            preset=quality.preset,
            crf=quality.clip_crf,
            should_cancel=cancel,
        )
        clip_files.append(dst)
        progress(0.6 * (i + 1) / len(tl.video), f"clip {i + 1}/{len(tl.video)}")

    # 2. concat
    silent = work / "silent.mp4"
    media.concat_videos(clip_files, silent, list_file=work / "clips.txt", should_cancel=cancel)
    for f in clip_files:
        f.unlink()
    progress(0.65, "joined")

    # 3. mixdown
    if not tl.narration and not (audio is not None and tl.source_audio) and not tl.bgm:
        raise RenderError("timeline has no sound at all")
    mix_file = work / "mix.wav"
    report = mix_timeline(
        tl, narration_dir, mix_file, film_audio=audio, bgm_dir=bgm_dir, mix=mix, cancel=cancel
    )
    progress(0.8, "mixed")

    # 4. final encode with subtitles
    subs: Path | None = None
    if tl.subtitles:
        subs = out_dir / SUBTITLES_FILE
        subs.write_text(build_ass(tl.subtitles, fps, out.width, out.height, font), encoding="utf-8")
    media.encode_final(
        silent,
        mix_file,
        out_dir / FINAL_FILE,
        subtitles=subs,
        duration_ms=total_ms,
        preset=quality.preset,
        crf=quality.final_crf,
        on_progress=lambda f: progress(0.8 + 0.2 * f, "encoding"),
        should_cancel=cancel,
    )
    return report


def mix_timeline(
    tl: Timeline,
    narration_dir: Path,
    dst: Path,
    *,
    film_audio: Path | None,
    bgm_dir: Path | None,
    mix: MixSettings,
    cancel: Callable[[], bool] | None = None,
) -> MixReport:
    """All of the timeline's sound in one wav of exactly the programme's length: narration,
    the film's sound under each clip, and music ducked under the narration (ARCHITECTURE §7.5)."""
    fps, rate, ch = tl.output.fps, mix.sample_rate, mix.channels
    total = frame_to_sample(tl.duration_frames, fps, rate)

    def at(frame: int) -> int:
        return frame_to_sample(frame, fps, rate)

    def decode(path: Path, seek_ms: int = 0, duration_ms: int | None = None) -> Audio:
        if not path.is_file():
            raise RenderError(f"audio file missing: {path}")
        return audio_io.decode_audio(
            path,
            sample_rate=rate,
            channels=ch,
            seek_ms=seek_ms,
            duration_ms=duration_ms,
            should_cancel=cancel,
        )

    voice = [Placement(decode(narration_dir / n.file), at(n.f0), n.gain_db) for n in tl.narration]
    spoken = [(p.start, p.start + len(p.audio)) for p in voice]
    layers = [Layer(voice)]
    if film_audio is not None and tl.source_audio:
        film = [
            Placement(
                decode(film_audio, s.src_in_ms, frames_to_ms(s.f1 - s.f0, fps)), at(s.f0), s.gain_db
            )
            for s in tl.source_audio
        ]
        layers.append(Layer(film))
    for b in tl.bgm:
        if bgm_dir is None:
            raise RenderError("the timeline has background music but no music folder was given")
        music = Placement(decode(bgm_dir / b.file, 0, frames_to_ms(b.f1 - b.f0, fps)), at(b.f0))
        layers.append(
            Layer(
                [Placement(music.audio, music.start, b.gain_db)],
                duck_db=b.duck.depth_db if b.duck else None,
                duck_under=spoken,
            )
        )
    signal, report = mixdown(total, layers, mix)
    audio_io.write_wav(dst, signal, rate)
    return report
