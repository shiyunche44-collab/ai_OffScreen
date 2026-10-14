"""Audio in and out of numpy for the mixdown (ffmpeg decodes and encodes; see `algo/mixing`).

Decoding goes through a raw float32 file in a scratch directory, so it uses the same
`run_ffmpeg` (cancel, errors) as every other step."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from offscreen.algo.mixing import RATE, Audio
from offscreen.media.ffmpeg import run_ffmpeg
from offscreen.media.render import Cancel


def decode_stereo(
    path: Path,
    scratch: Path,
    *,
    seek_ms: int = 0,
    duration_ms: int | None = None,
    should_cancel: Cancel = None,
) -> Audio:
    """The file's first audio stream (or `duration_ms` of it from `seek_ms`) as 48 kHz stereo
    float32. `scratch` is a file path for the intermediate raw data; it is removed."""
    args: list[str] = []
    if seek_ms:
        args += ["-ss", f"{seek_ms / 1000:.3f}"]
    if duration_ms is not None:
        args += ["-t", f"{duration_ms / 1000:.3f}"]
    args += [
        "-i", str(path), "-map", "0:a:0", "-vn",
        "-f", "f32le", "-ar", str(RATE), "-ac", "2", str(scratch),
    ]  # fmt: skip
    run_ffmpeg(args, should_cancel=should_cancel)
    try:
        return np.fromfile(scratch, dtype="<f4").reshape(-1, 2)
    finally:
        scratch.unlink(missing_ok=True)


def fit_length(x: Audio, n: int) -> Audio:
    """Exactly `n` samples: cut the end, or pad with silence."""
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - len(x), 2), dtype=np.float32)])


def write_wav(
    audio: Audio, dst: Path, scratch: Path, *, float_format: bool = True, cancel: Cancel = None
) -> None:
    """48 kHz stereo wav (32-bit float by default, so levels above full scale survive until
    the loudness step). `scratch` is a file path for the raw data; it is removed."""
    np.ascontiguousarray(audio, dtype="<f4").tofile(scratch)
    codec = "pcm_f32le" if float_format else "pcm_s16le"
    args = [
        "-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(scratch),
        "-c:a", codec, str(dst),
    ]  # fmt: skip
    try:
        run_ffmpeg(args, should_cancel=cancel)
    finally:
        scratch.unlink(missing_ok=True)
