from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from offscreen.algo.loudness import LoudnessSettings, meets_target
from offscreen.media.audio import decode_audio, write_wav
from offscreen.media.ffmpeg import FFmpegError
from offscreen.media.loudness import (
    measure_loudness,
    normalize_loudness,
    parse_loudnorm,
    write_raw,
)

RATE = 48_000


def speechlike(seconds: float, level: float, rate: int = RATE, seed: int = 1) -> np.ndarray:
    """Bursts of noise with pauses: a crest factor like speech's, so the true-peak ceiling is
    what limits the gain (as with real narration)."""
    rng = np.random.default_rng(seed)
    n = round(seconds * rate)
    env = (np.sin(np.arange(n) / rate * 2 * np.pi * 3) > -0.2).astype(np.float32)
    x = rng.standard_normal(n).astype(np.float32) * env * level
    return np.repeat(x[:, None], 2, axis=1)


def test_parse_loudnorm_takes_the_last_block() -> None:
    log = 'x\n{\n"input_i" : "-20.00", "input_tp" : "-inf"\n}\n[a] {\n"input_i" : "-9.50"\n}\n'
    assert parse_loudnorm(log) == {"input_i": "-9.50"}
    with pytest.raises(FFmpegError):
        parse_loudnorm("no stats here")


def test_measure_a_known_level(tmp_path: Path) -> None:
    """A 1 kHz sine with peaks at -20 dBFS on both channels reads -20 LUFS (the K-weighting is
    ~0 dB at 1 kHz, and two channels add 3 dB to a sine's -3 dB mean square)."""
    t = np.arange(4 * RATE) / RATE
    x = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    write_wav(tmp_path / "s.wav", np.repeat(x[:, None], 2, axis=1), RATE)
    m = measure_loudness(tmp_path / "s.wav")
    assert m.integrated == pytest.approx(-20.0, abs=0.5)
    assert m.true_peak == pytest.approx(-20.0, abs=0.3)
    assert not m.silent


def test_silence_is_measured_and_copied(tmp_path: Path) -> None:
    raw = write_raw(tmp_path / "z.f32", np.zeros((RATE, 2), dtype=np.float32), RATE)
    assert measure_loudness(tmp_path / "z.f32", raw=raw).silent
    out = normalize_loudness(
        tmp_path / "z.f32", tmp_path / "z.wav", LoudnessSettings(), sample_rate=RATE, raw=raw
    )
    assert out.mode == "none" and out.output.silent
    assert np.all(decode_audio(tmp_path / "z.wav", sample_rate=RATE, channels=2) == 0)


@pytest.mark.parametrize("level", [0.02, 0.3, 2.5])  # very quiet, loud, beyond full scale
def test_two_passes_reach_the_target(tmp_path: Path, level: float) -> None:
    settings = LoudnessSettings()
    raw = write_raw(tmp_path / "m.f32", speechlike(12.0, level), RATE)
    out = normalize_loudness(
        tmp_path / "m.f32", tmp_path / "m.wav", settings, sample_rate=RATE, raw=raw
    )
    assert out.mode in ("linear", "dynamic")
    final = measure_loudness(tmp_path / "m.wav")
    assert meets_target(final, settings), final
    audio = decode_audio(tmp_path / "m.wav", sample_rate=RATE, channels=2)
    assert len(audio) == 12 * RATE  # loudnorm works at 192 kHz; the rate comes back
    assert np.abs(audio).max() < 1.0  # nothing clipped


def test_a_programme_with_headroom_is_normalized_linearly(tmp_path: Path) -> None:
    """A steady tone has no peaks to tame: one gain for the whole programme."""
    t = np.arange(10 * RATE) / RATE
    tone = (0.01 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    raw = write_raw(tmp_path / "t.f32", np.repeat(tone[:, None], 2, axis=1), RATE)
    out = normalize_loudness(
        tmp_path / "t.f32", tmp_path / "t.wav", LoudnessSettings(), sample_rate=RATE, raw=raw
    )
    assert out.mode == "linear"
    assert measure_loudness(tmp_path / "t.wav").integrated == pytest.approx(-14.0, abs=0.5)


def test_write_wav_and_measure_agree_on_mono(tmp_path: Path) -> None:
    x = speechlike(5.0, 0.1)[:, :1]
    write_wav(tmp_path / "mono.wav", x, 32_000)
    assert -40 < measure_loudness(tmp_path / "mono.wav").integrated < -10
