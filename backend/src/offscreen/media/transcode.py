"""Derived media files: browser-friendly proxy video, analysis audio, thumbnails."""

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


THUMB_HEIGHT = 360
GRAY_WIDTH = 320
"""Width of the grayscale copy of a keyframe that quality metrics are measured on."""
RGB_SIZE = (64, 36)
"""Size of the small colour copy of a keyframe that colour histograms are taken from."""


def extract_frame(
    src: Path,
    dst: Path,
    *,
    at_ms: int,
    height: int = THUMB_HEIGHT,
    gray_to: Path | None = None,
    gray_width: int = GRAY_WIDTH,
    rgb_to: Path | None = None,
    rgb_size: tuple[int, int] = RGB_SIZE,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """One jpeg frame at `at_ms`, at most `height` pixels tall (never upscaled).

    `gray_to` additionally gets the same frame as raw 8-bit grayscale, `gray_width` pixels wide
    (height follows the aspect ratio), `rgb_to` as raw rgb24 squeezed to `rgb_size` (width,
    height): the inputs of the image metrics, produced in the same decode."""
    jpeg_scale = f"scale=-2:trunc(min({height}\\,ih)/2)*2"
    if gray_to is None and rgb_to is None:
        args = [
            "-ss", f"{at_ms / 1000:.3f}", "-i", str(src), "-map", "0:v:0", "-frames:v", "1",
            "-vf", jpeg_scale, "-q:v", "3", str(dst),
        ]  # fmt: skip
    else:
        # One decode, several outputs: (filter chain, destination, extra output options).
        outputs: list[tuple[str, Path, list[str]]] = [(jpeg_scale, dst, ["-q:v", "3"])]
        if gray_to is not None:
            outputs.append((f"scale={gray_width}:-2,format=gray", gray_to, ["-f", "rawvideo"]))
        if rgb_to is not None:
            rgb = f"scale={rgb_size[0]}:{rgb_size[1]},format=rgb24"
            outputs.append((rgb, rgb_to, ["-f", "rawvideo"]))
        graph = f"[0:v:0]split={len(outputs)}" + "".join(f"[s{i}]" for i in range(len(outputs)))
        graph += "".join(f";[s{i}]{flt}[o{i}]" for i, (flt, _, _) in enumerate(outputs))
        args = ["-ss", f"{at_ms / 1000:.3f}", "-i", str(src), "-filter_complex", graph]
        for i, (_, path, extra) in enumerate(outputs):
            args += ["-map", f"[o{i}]", "-frames:v", "1", *extra, str(path)]
    run_ffmpeg(args, should_cancel=should_cancel)


def make_sprite_sheets(
    frames: list[Path],
    out_pattern: Path,
    *,
    tile_width: int,
    tile_height: int,
    columns: int,
    rows: int,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    """Pack `frames` (in order) into jpeg sheets of `columns x rows` tiles, written to
    `out_pattern` (a `%03d` pattern, numbered from 1). Frames keep their aspect ratio inside a
    tile (letterboxed); the last sheet holds the remainder."""
    if not frames:
        raise ValueError("no frames to pack")
    listing = out_pattern.parent / ".sprite_frames.txt"
    listing.write_text(
        "".join(f"file '{f.resolve()}'\nduration 1\n" for f in frames), encoding="utf-8"
    )
    fit = (
        f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
        f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2"
    )
    try:
        run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-an",
                "-fps_mode",
                "passthrough",
                "-vf",
                f"{fit},tile={columns}x{rows}",
                "-q:v",
                "4",
                str(out_pattern),
            ],
            should_cancel=should_cancel,
        )
    finally:
        listing.unlink(missing_ok=True)
