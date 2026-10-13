"""Loudness decisions of the renderer (ARCHITECTURE §7.5 ②, M6-07). Pure: the measurements
(EBU R128 via ffmpeg loudnorm) are taken by `media/loudness.py`.

Two steps: each narration file is first brought to one common loudness, so no sentence comes
out louder than the next whatever the engine did; then the whole mixdown is normalized to the
programme target with a true-peak ceiling (two-pass loudnorm)."""

from __future__ import annotations

import math
from dataclasses import dataclass

SILENCE_LUFS = -70.0
"""Below the R128 absolute gate: nothing to measure, nothing to normalize."""


@dataclass(frozen=True)
class Loudness:
    integrated: float
    """LUFS (-inf for silence)."""
    true_peak: float
    """dBTP."""
    lra: float
    """Loudness range, LU."""
    threshold: float
    """Relative gate of the integrated measurement, LUFS."""

    @property
    def silent(self) -> bool:
        return not math.isfinite(self.integrated) or self.integrated < SILENCE_LUFS


@dataclass(frozen=True)
class LoudnessSettings:
    target_lufs: float = -14.0
    """Integrated loudness of the finished programme (short-video platforms)."""
    tolerance_lu: float = 1.0
    max_true_peak_db: float = -1.0
    """The ceiling the finished file must respect."""
    true_peak_db: float = -1.5
    """What the normalizer aims at: below the ceiling, so the AAC encoder's overshoot fits."""
    narration_lufs: float = -16.0
    """Every narration file is brought to this before mixing."""
    max_narration_gain_db: float = 12.0
    """No narration file is raised or lowered by more (a nearly silent one stays quiet)."""

    def __post_init__(self) -> None:
        if not -70 <= self.target_lufs <= -5 or not -70 <= self.narration_lufs <= -5:
            raise ValueError("loudness targets must be within -70..-5 LUFS")
        if not -9 <= self.true_peak_db <= 0:
            raise ValueError("true_peak_db must be within -9..0 dBTP")
        if self.true_peak_db > self.max_true_peak_db:
            raise ValueError("true_peak_db must not be above max_true_peak_db")
        if self.tolerance_lu <= 0 or self.max_narration_gain_db < 0:
            raise ValueError("tolerance and max gain must be positive")


def narration_gain_db(measured: Loudness, settings: LoudnessSettings) -> float:
    """Gain bringing one narration file to `narration_lufs` (0 for silence), clamped."""
    if measured.silent:
        return 0.0
    gain = settings.narration_lufs - measured.integrated
    cap = settings.max_narration_gain_db
    return round(min(cap, max(-cap, gain)), 2)


def loudnorm_lra(measured: Loudness, default: float = 11.0) -> float:
    """The loudness-range target for the second loudnorm pass: never below what was measured,
    so a wide-range programme is not squeezed (and linear mode stays possible)."""
    lra = measured.lra if math.isfinite(measured.lra) else 0.0
    return round(min(50.0, max(default, lra + 1.0)), 1)


def meets_target(final: Loudness, settings: LoudnessSettings) -> bool:
    """Exit criterion of M6: integrated loudness within tolerance, true peak under the ceiling."""
    if final.silent:
        return False
    return (
        abs(final.integrated - settings.target_lufs) <= settings.tolerance_lu
        and final.true_peak <= settings.max_true_peak_db
    )
