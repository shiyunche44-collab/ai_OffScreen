"""EBU R128 measurement and two-pass normalization with ffmpeg's loudnorm filter (M6-07).
The decisions (targets, per-file gains, the pass/fail check) are in `algo/loudness.py`."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from offscreen.algo.loudness import Loudness, LoudnessSettings, loudnorm_lra
from offscreen.algo.mixing import Audio
from offscreen.media.ffmpeg import FFmpegError, run_ffmpeg_log

Cancel = Callable[[], bool] | None


@dataclass(frozen=True)
class RawAudio:
    """Headerless little-endian float32 interleaved samples (the unclipped mixdown)."""

    sample_rate: int
    channels: int

    def input_args(self) -> list[str]:
        return ["-f", "f32le", "-ar", str(self.sample_rate), "-ac", str(self.channels)]


def write_raw(path: Path, audio: Audio, sample_rate: int) -> RawAudio:
    """Keep samples beyond full scale as they are: the normalizer brings them down."""
    np.ascontiguousarray(audio, dtype="<f4").tofile(path)
    return RawAudio(sample_rate=sample_rate, channels=audio.shape[1])


@dataclass(frozen=True)
class Normalized:
    measured: Loudness
    """The input, as measured by the first pass."""
    output: Loudness
    """What loudnorm reports for its output (before any later encoding)."""
    mode: str
    """"linear" (one gain for the whole programme), "dynamic" (loudnorm had to compress to
    keep the true-peak ceiling) or "none" (silence: copied as it was)."""


_JSON = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.S)


def parse_loudnorm(log: str) -> dict[str, str]:
    """The last statistics block loudnorm printed (`print_format=json`)."""
    found = _JSON.findall(log)
    if not found:
        raise FFmpegError(
            "loudnorm printed no statistics", returncode=0, cmd=[], stderr=log[-2000:]
        )
    stats: dict[str, str] = json.loads(found[-1])
    return stats


def _num(text: str) -> float:
    return float(text)  # float() reads "-inf" and "inf" as well


def _loudness(stats: dict[str, str], side: str) -> Loudness:
    return Loudness(
        integrated=_num(stats[f"{side}_i"]),
        true_peak=_num(stats[f"{side}_tp"]),
        lra=_num(stats[f"{side}_lra"]),
        threshold=_num(stats[f"{side}_thresh"]),
    )


def _input(path: Path, raw: RawAudio | None) -> list[str]:
    return [*(raw.input_args() if raw else []), "-i", str(path), "-map", "0:a:0"]


def measure_loudness(
    path: Path, *, raw: RawAudio | None = None, should_cancel: Cancel = None
) -> Loudness:
    """Integrated loudness, true peak and loudness range of the first audio stream."""
    log = run_ffmpeg_log(
        [*_input(path, raw), "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        should_cancel=should_cancel,
    )
    return _loudness(parse_loudnorm(log), "input")


def _clamp(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return lo if x < 0 else hi
    return min(hi, max(lo, x))


def normalize_loudness(
    src: Path,
    dst: Path,
    settings: LoudnessSettings,
    *,
    sample_rate: int,
    raw: RawAudio | None = None,
    should_cancel: Cancel = None,
) -> Normalized:
    """Two passes: measure `src`, then write it to `dst` (16-bit wav at `sample_rate`)
    normalized to the programme target with the measured values (linear when the true-peak
    ceiling allows it)."""
    measured = measure_loudness(src, raw=raw, should_cancel=should_cancel)
    out_args = ["-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    if measured.silent:
        run_ffmpeg_log([*_input(src, raw), *out_args], should_cancel=should_cancel)
        return Normalized(measured, measured, "none")
    flt = (
        f"loudnorm=I={settings.target_lufs}:TP={settings.true_peak_db}"
        f":LRA={loudnorm_lra(measured)}"
        f":measured_I={_clamp(measured.integrated, -99, 0):.2f}"
        f":measured_TP={_clamp(measured.true_peak, -99, 99):.2f}"
        # loudnorm takes a measured LRA of 0 for "not measured" and then never goes linear
        f":measured_LRA={_clamp(measured.lra, 0.01, 99):.2f}"
        f":measured_thresh={_clamp(measured.threshold, -99, 0):.2f}"
        ":linear=true:print_format=json"
    )
    log = run_ffmpeg_log([*_input(src, raw), "-af", flt, *out_args], should_cancel=should_cancel)
    stats = parse_loudnorm(log)
    return Normalized(measured, _loudness(stats, "output"), stats.get("normalization_type", "?"))
