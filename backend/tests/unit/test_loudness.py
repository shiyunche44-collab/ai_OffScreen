from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.loudness import (
    MAX_BOOST_DB,
    MAX_CUT_DB,
    NARRATION_LUFS,
    PROGRAMME_LUFS,
    Measurement,
    meets_target,
    narration_gain_db,
)
from offscreen.media.audio import MIX_RATE, write_wav
from offscreen.media.ffmpeg import FFmpegError
from offscreen.media.loudness import measure, normalize, parse_loudnorm_json

LOG = """[Parsed_loudnorm_0 @ 0x1] 
{
	"input_i" : "-27.61",
	"input_tp" : "-4.47",
	"input_lra" : "18.06",
	"input_thresh" : "-39.20",
	"output_i" : "-16.58",
	"output_tp" : "-1.50",
	"output_lra" : "14.40",
	"output_thresh" : "-27.71",
	"normalization_type" : "dynamic",
	"target_offset" : "0.58"
}
"""


def test_parse_loudnorm_report() -> None:
    m = parse_loudnorm_json(LOG)
    assert (m.integrated_lufs, m.true_peak_dbtp, m.lra) == (-27.61, -4.47, 18.06)
    assert (m.threshold_lufs, m.target_offset_db) == (-39.2, 0.58)


def test_parse_silence_and_garbage() -> None:
    m = parse_loudnorm_json('{"input_i": "-inf", "input_tp": "-inf", "input_lra": "0.00"}')
    assert m.silent and math.isinf(m.integrated_lufs) and m.as_dict()["integrated_lufs"] == -99.0
    with pytest.raises(FFmpegError):
        parse_loudnorm_json("no json here")


def test_narration_gain_levels_and_bounds() -> None:
    assert narration_gain_db(-22.0) == pytest.approx(NARRATION_LUFS + 22.0)
    assert narration_gain_db(-10.0) == pytest.approx(-6.0)
    assert narration_gain_db(-60.0) == MAX_BOOST_DB  # never boosted into noise
    assert narration_gain_db(5.0, target=-16.0) == -MAX_CUT_DB
    assert narration_gain_db(float("-inf")) == 0.0 and narration_gain_db(-80.0) == 0.0


@given(st.floats(-69.0, 0.0))
def test_narration_gain_lands_on_target_when_unbounded(lufs: float) -> None:
    g = narration_gain_db(lufs)
    assert -MAX_CUT_DB <= g <= MAX_BOOST_DB
    if -MAX_CUT_DB < NARRATION_LUFS - lufs < MAX_BOOST_DB:
        assert lufs + g == pytest.approx(NARRATION_LUFS)


def test_meets_target() -> None:
    ok = Measurement(-14.4, -1.2, 6.0, -24.0)
    assert meets_target(ok)
    assert not meets_target(Measurement(-16.0, -1.5, 6.0, -26.0))  # too quiet
    assert not meets_target(Measurement(-14.0, -0.2, 6.0, -24.0))  # peak over the ceiling
    assert not meets_target(Measurement(float("-inf"), float("-inf"), 0.0, -70.0))


# ---- real ffmpeg --------------------------------------------------------------------------
def tone(seconds: float, amp: float) -> np.ndarray:
    t = np.arange(round(seconds * MIX_RATE)) / MIX_RATE
    # A few partials, so it is not a pure sine, and some quiet gaps like speech.
    x = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((220, 440, 880, 1760)))
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t) ** 2)
    x = (x / np.abs(x).max() * amp).astype(np.float32)
    return np.stack([x, x], axis=1)


def test_two_pass_normalization_hits_the_target(tmp_path: Path) -> None:
    src, dst = tmp_path / "quiet.wav", tmp_path / "norm.wav"
    write_wav(src, tone(12.0, 0.05))
    before = measure(src)
    assert before.integrated_lufs < -25  # clearly too quiet
    after = normalize(src, dst, measured=before)
    assert meets_target(after), after
    assert abs(after.integrated_lufs - PROGRAMME_LUFS) <= 1.0 and after.true_peak_dbtp <= -0.9


def test_loud_input_is_brought_down_and_kept_under_the_peak_ceiling(tmp_path: Path) -> None:
    src, dst = tmp_path / "loud.wav", tmp_path / "norm.wav"
    write_wav(src, tone(12.0, 0.99))
    after = normalize(src, dst, measured=measure(src))
    assert meets_target(after), after


def test_silence_is_reported_silent(tmp_path: Path) -> None:
    src = tmp_path / "silence.wav"
    write_wav(src, np.zeros((MIX_RATE * 2, 2), dtype=np.float32))
    assert measure(src).silent
