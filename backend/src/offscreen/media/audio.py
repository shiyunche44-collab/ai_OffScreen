"""Sample-level audio IO for the mixer: decode any file to float stereo, write the mix as wav."""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from offscreen.algo.mixing import Samples
from offscreen.media.ffmpeg import read_raw_audio

MIX_RATE = 48_000


def read_pcm(
    path: Path,
    *,
    rate: int = MIX_RATE,
    seek_ms: int = 0,
    duration_ms: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Samples:
    """`(n, 2)` float32 at `rate`; the file's first audio stream, mono upmixed to both sides."""
    raw = read_raw_audio(
        str(path), rate=rate, seek_ms=seek_ms, duration_ms=duration_ms, should_cancel=should_cancel
    )
    flat = np.frombuffer(raw[: len(raw) // 8 * 8], dtype="<f4")
    return flat.reshape(-1, 2).astype(np.float32, copy=True)


def write_wav(path: Path, samples: Samples, *, rate: int = MIX_RATE) -> int:
    """16-bit stereo wav. Returns how many samples had to be clipped (0 on a clean mix)."""
    scaled = np.round(samples.astype(np.float64) * 32768.0)
    clipped = int(np.count_nonzero((scaled > 32767) | (scaled < -32768)))
    pcm = np.clip(scaled, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return clipped
