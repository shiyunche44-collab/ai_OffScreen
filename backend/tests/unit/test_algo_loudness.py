from __future__ import annotations

import math

import pytest

from offscreen.algo.loudness import (
    Loudness,
    LoudnessSettings,
    loudnorm_lra,
    meets_target,
    narration_gain_db,
)

S = LoudnessSettings()


def lou(i: float, tp: float = -3.0, lra: float = 5.0) -> Loudness:
    return Loudness(integrated=i, true_peak=tp, lra=lra, threshold=i - 10)


def test_settings_are_validated() -> None:
    for bad in (
        {"target_lufs": 0.0},
        {"narration_lufs": -80.0},
        {"true_peak_db": 1.0},
        {"true_peak_db": -0.5, "max_true_peak_db": -1.0},
        {"tolerance_lu": 0.0},
    ):
        with pytest.raises(ValueError):
            LoudnessSettings(**bad)  # type: ignore[arg-type]


def test_silence() -> None:
    assert lou(-math.inf).silent and lou(-75.0).silent and not lou(-40.0).silent


def test_narration_gain_brings_every_file_to_one_level() -> None:
    assert narration_gain_db(lou(-20.0), S) == 4.0
    assert narration_gain_db(lou(-11.5), S) == -4.5
    assert narration_gain_db(lou(-40.0), S) == 12.0  # capped: a whisper stays quiet-ish
    assert narration_gain_db(lou(-2.0), S) == -12.0  # capped the other way too
    assert narration_gain_db(lou(-math.inf), S) == 0.0


def test_loudnorm_lra_never_squeezes_the_measured_range() -> None:
    assert loudnorm_lra(lou(-20, lra=3.0)) == 11.0
    assert loudnorm_lra(lou(-20, lra=18.2)) == 19.2
    assert loudnorm_lra(lou(-20, lra=70.0)) == 50.0
    assert loudnorm_lra(lou(-20, lra=math.nan)) == 11.0


def test_meets_target() -> None:
    assert meets_target(lou(-14.0, tp=-1.2), S)
    assert meets_target(lou(-13.0, tp=-1.0), S) and meets_target(lou(-15.0, tp=-3.0), S)
    assert not meets_target(lou(-12.9, tp=-2.0), S)
    assert not meets_target(lou(-15.1, tp=-2.0), S)
    assert not meets_target(lou(-14.0, tp=-0.9), S)
    assert not meets_target(lou(-math.inf), S)
