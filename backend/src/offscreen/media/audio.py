"""Audio files in and out of the mixer (`algo/mixing.py`): decode any file ffmpeg reads into
float32 samples, write the mixdown as a wav."""

from __future__ import annotations

import functools
import subprocess
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from offscreen.algo.mixing import Audio
from offscreen.media.ffmpeg import FFmpegError, read_ffmpeg_output


def decode_audio(
    path: Path,
    *,
    sample_rate: int,
    channels: int,
    seek_ms: int = 0,
    duration_ms: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Audio:
    """The first audio stream of `path` (from `seek_ms`, at most `duration_ms` of it), resampled
    to `sample_rate` with `channels` channels: shape `(samples, channels)`. When a duration is
    given the result is exactly that long (padded with silence if the file ends sooner).

    A mono file is copied to every channel at full level (ffmpeg's own upmix would put it
    3 dB lower on each side)."""
    mono = channels > 1 and audio_channels(path) == 1
    want_ch = 1 if mono else channels
    args: list[str] = []
    if seek_ms:
        args += ["-ss", f"{seek_ms / 1000:.3f}"]
    if duration_ms is not None:
        args += ["-t", f"{duration_ms / 1000:.3f}"]
    args += [
        "-i", str(path), "-map", "0:a:0", "-vn",
        "-ac", str(want_ch), "-ar", str(sample_rate),
        "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1",
    ]  # fmt: skip
    raw = read_ffmpeg_output(args, should_cancel=should_cancel)
    n = len(raw) // (4 * want_ch)
    audio = np.frombuffer(raw[: n * 4 * want_ch], dtype="<f4").reshape(n, want_ch)
    if mono:
        audio = np.repeat(audio, channels, axis=1)
    if duration_ms is None:
        return audio.astype(np.float32)
    want = round(duration_ms * sample_rate / 1000)
    out = np.zeros((want, channels), dtype=np.float32)
    out[: min(want, n)] = audio[:want]
    return out


@functools.lru_cache(maxsize=256)
def _channels_cached(path: str, mtime_ns: int, size: int) -> int:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=channels", "-of", "csv=p=0", path,
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )  # fmt: skip
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffprobe failed for {path}",
            returncode=proc.returncode,
            cmd=["ffprobe", path],
            stderr=proc.stderr,
        )
    text = proc.stdout.strip().split(",")[0]
    if not text.isdigit():
        raise FFmpegError(
            f"no audio stream in {path}", returncode=0, cmd=["ffprobe", path], stderr=""
        )
    return int(text)


def audio_channels(path: Path) -> int:
    """Channel count of the first audio stream (remembered while the file is unchanged: a film
    is cut into many pieces of sound)."""
    st = path.stat()
    return _channels_cached(str(path), st.st_mtime_ns, st.st_size)


def write_wav(path: Path, audio: Audio, sample_rate: int) -> None:
    """16-bit PCM wav; samples beyond full scale are clamped."""
    pcm = np.clip(np.rint(audio * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
