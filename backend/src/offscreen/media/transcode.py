"""Derived media files: browser-friendly proxy video and analysis audio."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from offscreen.domain.common import Rational
from offscreen.media.ffmpeg import ProgressFn, run_ffmpeg

PROXY_HEIGHT = 540
GOP_SECONDS = 0.5


def gop_frames(fps: Rational, seconds: float = GOP_SECONDS) -> int:
    return max(1, round(fps.as_float() * seconds))


def make_proxy(
    src: Path,
    dst: Path,
    *,
    fps: Rational,
    duration_ms: int,
    has_audio: bool,
    height: int = PROXY_HEIGHT,
    on_progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """H.264 + AAC mp4 at most `height` pixels tall (never upscaled), keyframe every 0.5 s
    on a fixed grid, moov atom up front, so the browser can seek and scrub smoothly."""
    gop = gop_frames(fps)
    scale = f"scale=-2:trunc(min({height}\\,ih)/2)*2"
    args = ["-i", str(src), "-map", "0:v:0"]
    if has_audio:
        args += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k", "-ac", "2"]
    else:
        args += ["-an"]
    args += [
        "-vf", scale,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-movflags", "+faststart",
        str(dst),
    ]  # fmt: skip
    run_ffmpeg(args, duration_ms=duration_ms, on_progress=on_progress, should_cancel=should_cancel)


def extract_audio(
    src: Path,
    dst: Path,
    *,
    stream_index: int,
    sample_rate: int,
    channels: int,
    duration_ms: int,
    on_progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """16-bit PCM wav of one source audio stream (absolute stream index)."""
    args = [
        "-i", str(src), "-map", f"0:{stream_index}", "-vn",
        "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_s16le",
        str(dst),
    ]  # fmt: skip
    run_ffmpeg(args, duration_ms=duration_ms, on_progress=on_progress, should_cancel=should_cancel)
