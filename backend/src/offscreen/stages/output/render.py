"""render: timeline -> final.mp4 (T5, deterministic: no model is called).

Reads only the compile artifact (timeline + the narration files beside it) and the source
film the timeline's items point to. Steps (ARCHITECTURE §7.5), all in a scratch directory
that is removed afterwards:
1. each video item becomes a silent clip of exactly `f1 - f0` frames in one common encoding;
2. the clips are joined without re-encoding;
3. all sound (narration, the film's own audio under it, music ducked under the narration) is
   mixed as a whole, sample by sample (`algo/mixing.py`), each narration file first brought
   to one loudness; the mixdown is then normalized to the programme target (-14 LUFS, true
   peak under -1 dBTP) in two loudnorm passes (`algo/loudness.py`);
4. the video is re-encoded once with the subtitles burned in, with the mixdown as AAC;
5. the finished file is measured, and everything measured is written to `loudness.json`.

v0 keeps the source frame size (layout "keep") and takes the film's first audio stream. The
timeline's music is only rendered when a music folder is given; nothing compiles music into a
timeline before M6-08."""

from __future__ import annotations

import dataclasses
import json
import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from offscreen.algo.ass import FONT, build_ass
from offscreen.algo.loudness import Loudness, LoudnessSettings, meets_target, narration_gain_db
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
from offscreen.media import loudness as loud
from offscreen.media import render as media
from offscreen.media.ffmpeg import FFmpegCanceled
from offscreen.media.loudness import Normalized
from offscreen.stages.output.compile import TIMELINE_FILE
from offscreen.store.files import atomic_write_bytes
from offscreen.store.repos import AssetRepo

FINAL_FILE = "final.mp4"
LOUDNESS_FILE = "loudness.json"
DEFAULT_MIX = MixSettings()
DEFAULT_LOUDNESS = LoudnessSettings()
SUBTITLES_FILE = "subtitles.ass"


class RenderError(RuntimeError):
    pass


def frames_to_ms(frames: int, fps: Rational) -> int:
    return round(frames * 1000 * fps.den / fps.num)


class RenderStage(Stage):
    name = "output.render"
    version = 3  # 2: numpy mixdown (edge fades, music ducked under the narration)
    # 3: narration leveled per file, programme normalized (two-pass loudnorm), loudness.json
    lane: Lane = "cpu"

    def __init__(
        self,
        assets: AssetRepo,
        font: str = FONT,
        mix: MixSettings = DEFAULT_MIX,
        loudness: LoudnessSettings = DEFAULT_LOUDNESS,
    ) -> None:
        self.assets = assets
        self.font = font
        self.mix = mix
        self.loudness = loudness

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
            "loudness": dataclasses.asdict(self.loudness),
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
                    loudness=self.loudness,
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
                "mix_clipped_samples": report.mix.clipped,
                "loudness_lufs": _finite(report.final.integrated),
                "true_peak_db": _finite(report.final.true_peak),
                "loudness_mode": report.normalized.mode,
                "loudness_within_target": report.within_target,
            }
        )


@dataclass(frozen=True)
class Quality:
    """Encoder settings: the final video's, or a preview's (fast and small)."""

    preset: str = media.PRESET
    clip_crf: int = media.CLIP_CRF
    final_crf: int = media.FINAL_CRF


DEFAULT_QUALITY = Quality()


@dataclass(frozen=True)
class SoundReport:
    """Written as `loudness.json` beside the film: how loud every part was and what came out."""

    settings: LoudnessSettings
    narration: list[NarrationLevel]
    mix: MixReport
    normalized: Normalized
    final: Loudness
    within_target: bool

    def to_json(self) -> dict[str, Any]:
        def lou(x: Loudness) -> dict[str, float | None]:
            return {k: _finite(v) for k, v in dataclasses.asdict(x).items()}

        return {
            "target": dataclasses.asdict(self.settings),
            "narration": [
                {"seg": n.seg, "file": n.file, "measured": lou(n.measured), "gain_db": n.gain_db}
                for n in self.narration
            ],
            "mix": {"peak": round(self.mix.peak, 4), "clipped_samples": self.mix.clipped},
            "normalization": {
                "mode": self.normalized.mode,
                "input": lou(self.normalized.measured),
                "output": lou(self.normalized.output),
            },
            "final": lou(self.final),
            "within_target": self.within_target,
        }


def _finite(x: float) -> float | None:
    """JSON has no infinities: silence is reported as null."""
    return round(x, 2) if math.isfinite(x) else None


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
    loudness: LoudnessSettings = DEFAULT_LOUDNESS,
    bgm_dir: Path | None = None,
    progress: Callable[[float, str], None] = lambda _f, _m: None,
    should_cancel: Callable[[], bool] | None = None,
) -> SoundReport:
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

    # 3. mixdown, then the whole programme to the loudness target (two passes)
    if not tl.narration and not (audio is not None and tl.source_audio) and not tl.bgm:
        raise RenderError("timeline has no sound at all")
    mixed = mix_timeline(
        tl,
        narration_dir,
        film_audio=audio,
        bgm_dir=bgm_dir,
        mix=mix,
        loudness=loudness,
        cancel=cancel,
    )
    progress(0.72, "mixed")
    raw_file, mix_file = work / "mix.f32", work / "mix.wav"
    raw = loud.write_raw(raw_file, mixed.audio, mix.sample_rate)
    mixed_levels, mix_report = mixed.narration, mixed.report
    del mixed  # the whole programme's samples: free them before encoding
    normalized = loud.normalize_loudness(
        raw_file, mix_file, loudness, sample_rate=mix.sample_rate, raw=raw, should_cancel=cancel
    )
    raw_file.unlink()
    progress(0.8, "loudness")

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
        on_progress=lambda f: progress(0.8 + 0.19 * f, "encoding"),
        should_cancel=cancel,
    )

    # 5. what the listener gets: measured on the finished file (after AAC)
    final = loud.measure_loudness(out_dir / FINAL_FILE, should_cancel=cancel)
    sound = SoundReport(
        settings=loudness,
        narration=mixed_levels,
        mix=mix_report,
        normalized=normalized,
        final=final,
        within_target=meets_target(final, loudness),
    )
    report_json = json.dumps(sound.to_json(), ensure_ascii=False, sort_keys=True, indent=2)
    atomic_write_bytes(out_dir / LOUDNESS_FILE, report_json.encode("utf-8"))
    return sound


@dataclass(frozen=True)
class NarrationLevel:
    seg: str
    file: str
    measured: Loudness
    gain_db: float


@dataclass(frozen=True)
class Mixdown:
    audio: Audio
    report: MixReport
    narration: list[NarrationLevel]


def mix_timeline(
    tl: Timeline,
    narration_dir: Path,
    *,
    film_audio: Path | None,
    bgm_dir: Path | None,
    mix: MixSettings,
    loudness: LoudnessSettings | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Mixdown:
    """All of the timeline's sound, exactly the programme's length: narration (each file
    brought to `loudness.narration_lufs` first, unless `loudness` is None), the film's sound
    under each clip, and music ducked under the narration (ARCHITECTURE §7.5)."""
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

    voice: list[Placement] = []
    levels: list[NarrationLevel] = []
    for n in tl.narration:
        path = narration_dir / n.file
        samples = decode(path)
        gain = n.gain_db
        if loudness is not None:
            measured = loud.measure_loudness(path, should_cancel=cancel)
            level = narration_gain_db(measured, loudness)
            levels.append(NarrationLevel(n.seg, n.file, measured, level))
            gain += level
        voice.append(Placement(samples, at(n.f0), gain))
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
    return Mixdown(signal, report, levels)
