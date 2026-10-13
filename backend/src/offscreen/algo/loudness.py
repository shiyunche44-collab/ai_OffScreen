"""Loudness targets and decisions (ARCHITECTURE §7.5). Pure: the measuring itself is ffmpeg's
(`media/loudness.py`); this module decides what to do with the numbers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

PROGRAMME_LUFS = -14.0
"""Short-video platforms' usual target for the whole programme."""
TRUE_PEAK_DBTP = -1.0
LOUDNESS_RANGE = 11.0
NARRATION_LUFS = -16.0
"""Each narration clip is levelled to this before mixing, so segments sound equally loud
whatever the voice or speed; the programme pass then brings the sum to `PROGRAMME_LUFS`."""
MAX_BOOST_DB = 12.0
MAX_CUT_DB = 18.0
LUFS_TOLERANCE = 1.0
SILENCE_LUFS = -70.0
"""At or below this the clip is (nearly) silent and is left alone."""


@dataclass(frozen=True)
class Measurement:
    """What ffmpeg's loudnorm reports about a file (EBU R128 / ITU-R BS.1770)."""

    integrated_lufs: float
    true_peak_dbtp: float
    lra: float
    threshold_lufs: float
    target_offset_db: float = 0.0

    @property
    def silent(self) -> bool:
        return not math.isfinite(self.integrated_lufs) or self.integrated_lufs <= SILENCE_LUFS

    def as_dict(self) -> dict[str, float]:
        return {k: (v if math.isfinite(v) else -99.0) for k, v in asdict(self).items()}


def narration_gain_db(
    measured_lufs: float,
    *,
    target: float = NARRATION_LUFS,
    max_boost_db: float = MAX_BOOST_DB,
    max_cut_db: float = MAX_CUT_DB,
) -> float:
    """The gain that brings a clip measured at `measured_lufs` to `target`, bounded so a quiet
    take is not boosted into noise. Silent or unmeasurable clips get 0."""
    if not math.isfinite(measured_lufs) or measured_lufs <= SILENCE_LUFS:
        return 0.0
    return min(max(target - measured_lufs, -max_cut_db), max_boost_db)


def meets_target(
    m: Measurement,
    *,
    target: float = PROGRAMME_LUFS,
    tolerance: float = LUFS_TOLERANCE,
    true_peak: float = TRUE_PEAK_DBTP,
) -> bool:
    """Exit criterion of M6: within `tolerance` LU of the target and under the true-peak ceiling
    (a small allowance for the encoder's rounding)."""
    return (
        not m.silent
        and abs(m.integrated_lufs - target) <= tolerance
        and m.true_peak_dbtp <= true_peak + 0.1
    )
