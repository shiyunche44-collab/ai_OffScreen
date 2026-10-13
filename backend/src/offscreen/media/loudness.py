"""Loudness measurement and the two-pass normalization (ffmpeg `loudnorm`)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from offscreen.algo.loudness import (
    LOUDNESS_RANGE,
    PROGRAMME_LUFS,
    TRUE_PEAK_DBTP,
    Measurement,
)
from offscreen.media.audio import MIX_RATE
from offscreen.media.ffmpeg import FFmpegError, capture_ffmpeg_log

_JSON = re.compile(r"\{[^{}]*\}", re.S)


def _num(v: object) -> float:
    try:
        return float(str(v))
    except ValueError:
        return float("-inf")


def parse_loudnorm_json(log: str) -> Measurement:
    """The JSON block loudnorm prints at the end of its run (values are strings; `-inf` for
    silence)."""
    blocks = _JSON.findall(log)
    for text in reversed(blocks):
        try:
            d = json.loads(text)
        except ValueError:
            continue
        if "input_i" in d:
            return Measurement(
                integrated_lufs=_num(d["input_i"]),
                true_peak_dbtp=_num(d["input_tp"]),
                lra=_num(d.get("input_lra", 0)),
                threshold_lufs=_num(d.get("input_thresh", -70)),
                target_offset_db=_num(d.get("target_offset", 0)),
            )
    raise FFmpegError(
        "no loudnorm report in the ffmpeg log", returncode=0, cmd=[], stderr=log[-2000:]
    )


def _filter(target: float, true_peak: float, lra: float, measured: Measurement | None) -> str:
    f = f"loudnorm=I={target}:TP={true_peak}:LRA={lra}"
    if measured is not None:
        f += (
            f":measured_I={measured.integrated_lufs}:measured_TP={measured.true_peak_dbtp}"
            f":measured_LRA={measured.lra}:measured_thresh={measured.threshold_lufs}"
            f":offset={measured.target_offset_db}:linear=true"
        )
    return f + ":print_format=json"


def measure(
    path: Path,
    *,
    target: float = PROGRAMME_LUFS,
    true_peak: float = TRUE_PEAK_DBTP,
    lra: float = LOUDNESS_RANGE,
) -> Measurement:
    """Integrated loudness, true peak and range of the file's first audio stream."""
    log = capture_ffmpeg_log(
        [
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            _filter(target, true_peak, lra, None),
            "-f",
            "null",
            "-",
        ]
    )
    return parse_loudnorm_json(log)


def normalize(
    src: Path,
    dst: Path,
    *,
    measured: Measurement,
    target: float = PROGRAMME_LUFS,
    true_peak: float = TRUE_PEAK_DBTP,
    lra: float = LOUDNESS_RANGE,
) -> Measurement:
    """Second pass: apply loudnorm with the first pass' numbers (a linear gain when the range
    allows it, so the dynamics are kept) and write a 48 kHz stereo wav. Returns the measurement
    of the result."""
    capture_ffmpeg_log(
        [
            "-i",
            str(src),
            "-map",
            "0:a:0",
            "-af",
            _filter(target, true_peak, lra, measured),
            "-ar",
            str(MIX_RATE),
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    return measure(dst, target=target, true_peak=true_peak, lra=lra)
