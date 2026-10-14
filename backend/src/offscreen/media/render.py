"""ffmpeg steps of the renderer (ARCHITECTURE §7.5): clips, concat, mixdown, final encode.

Every step is a thin function over one ffmpeg run, so each can be checked on its own. The
intermediate video files share one encoding (H.264, yuv420p, fixed frame rate), which is what
lets `concat_videos` join them without re-encoding."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from offscreen.domain.common import Rational
from offscreen.media.ffmpeg import ProgressFn, run_ffmpeg

CLIP_CRF = 16
FINAL_CRF = 18
PRESET = "veryfast"
AUDIO_BITRATE = "192k"

Cancel = Callable[[], bool] | None


def even(n: int) -> int:
    return max(2, n - n % 2)


def clip_filter(fps: Rational, width: int, height: int, speed: float, frames: int) -> str:
    """Video filter of one clip: retime, fix frame rate and size, and keep the last frame on
    screen if the source ran out early (the caller then cuts at exactly `frames`)."""
    parts = []
    if speed != 1.0:
        parts.append(f"setpts=PTS/{speed:.6f}")
    parts += [
        f"fps={fps.num}/{fps.den}",
        f"scale={even(width)}:{even(height)}:flags=lanczos",
        "setsar=1",
        "format=yuv420p",
        f"tpad=stop_mode=clone:stop={frames}",
    ]
    return ",".join(parts)


def render_clip(
    src: Path,
    dst: Path,
    *,
    src_in_ms: int,
    frames: int,
    fps: Rational,
    width: int,
    height: int,
    speed: float = 1.0,
    preset: str = PRESET,
    crf: int = CLIP_CRF,
    should_cancel: Cancel = None,
) -> None:
    """Exactly `frames` silent frames of `src` from `src_in_ms`, in the common encoding."""
    if frames < 1:
        raise ValueError("a clip needs at least one frame")
    args = [
        "-ss", f"{src_in_ms / 1000:.3f}", "-i", str(src),
        "-map", "0:v:0", "-an",
        "-vf", clip_filter(fps, width, height, speed, frames),
        "-frames:v", str(frames), "-r", f"{fps.num}/{fps.den}",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-g", "48", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
        str(dst),
    ]  # fmt: skip
    run_ffmpeg(args, should_cancel=should_cancel)


def _concat_quote(path: Path) -> str:
    """A path as the concat demuxer wants it: single-quoted, quotes inside as '\\''."""
    return "'" + str(path.resolve()).replace("'", "'\\''") + "'"


def concat_videos(
    parts: Sequence[Path], dst: Path, *, list_file: Path, should_cancel: Cancel = None
) -> None:
    """Join same-encoding clips without re-encoding."""
    if not parts:
        raise ValueError("nothing to concatenate")
    list_file.write_text("".join(f"file {_concat_quote(p)}\n" for p in parts), encoding="utf-8")
    run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(dst)],
        should_cancel=should_cancel,
    )


_FILTER_SPECIAL = re.compile(r"([\\':,\[\]])")


def escape_filter_value(path: str) -> str:
    """Make a path safe inside an ffmpeg filter argument."""
    return _FILTER_SPECIAL.sub(r"\\\1", path)


def encode_final(
    video: Path,
    audio: Path,
    dst: Path,
    *,
    subtitles: Path | None,
    duration_ms: int,
    preset: str = PRESET,
    crf: int = FINAL_CRF,
    on_progress: ProgressFn | None = None,
    should_cancel: Cancel = None,
) -> None:
    """Final mp4: the video re-encoded with the subtitles burned in (or copied when there are
    none), the mixdown as AAC, moov atom up front."""
    args = ["-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0"]
    if subtitles is not None:
        args += [
            "-vf", f"ass=filename={escape_filter_value(str(subtitles))}",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        ]  # fmt: skip
    else:
        args += ["-c:v", "copy"]
    args += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-movflags", "+faststart", str(dst)]
    run_ffmpeg(args, duration_ms=duration_ms, on_progress=on_progress, should_cancel=should_cancel)
