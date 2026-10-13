from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.mixing import (
    Layer,
    MixSettings,
    Placement,
    db_to_gain,
    duck_envelope,
    edge_fades,
    frame_to_sample,
    merge_spans,
    mix,
    seam_jumps,
)
from offscreen.domain.common import Rational

RATE = 1000  # 1 sample per ms keeps the arithmetic readable
S = MixSettings(
    sample_rate=RATE, channels=1, edge_fade_ms=10, duck_attack_ms=20, duck_release_ms=40
)


def const(n: int, value: float = 1.0, ch: int = 1) -> np.ndarray:
    return np.full((n, ch), value, dtype=np.float32)


def sine(n: int, hz: float, rate: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / rate
    return np.sin(2 * np.pi * hz * t + phase).astype(np.float32)[:, None]


# ---- settings and small helpers ------------------------------------------------------------
def test_settings_are_validated() -> None:
    for bad in ({"edge_fade_ms": 5}, {"edge_fade_ms": 25}, {"duck_attack_ms": -1}):
        with pytest.raises(ValueError):
            MixSettings(**bad)  # type: ignore[arg-type]


def test_frame_to_sample_has_no_drift() -> None:
    ntsc = Rational(num=24000, den=1001)
    assert frame_to_sample(0, ntsc, 48_000) == 0
    assert frame_to_sample(24_000, ntsc, 48_000) == 48_048_000  # 1001 s exactly
    assert frame_to_sample(25, Rational(num=25, den=1), 48_000) == 48_000


def test_edge_fades() -> None:
    g = edge_fades(100, 10)
    assert g.shape == (100,) and g.dtype == np.float32
    assert 0 < g[0] < 0.05 and 0 < g[-1] < 0.05 and g[0] == g[-1]
    assert np.all(np.diff(g[:10]) > 0) and np.all(g[10:90] == 1)
    short = edge_fades(6, 10)  # shorter than two fades: still rises and falls
    assert short.max() <= 1 and short[0] < short[2] and short[-1] < short[-3]
    assert edge_fades(1, 10).tolist() == [1.0] and len(edge_fades(0, 10)) == 0


def test_merge_spans() -> None:
    assert merge_spans([(50, 60), (0, 10), (5, 20), (30, 30)]) == [(0, 20), (50, 60)]
    assert merge_spans([(0, 10), (25, 30)], bridge=16) == [(0, 30)]
    assert merge_spans([(0, 10), (25, 30)], bridge=15) == [(0, 10), (25, 30)]


# ---- ducking --------------------------------------------------------------------------------
def test_music_goes_down_before_the_voice_and_comes_back_after() -> None:
    env = duck_envelope(300, [(100, 200)], -12.0, attack=20, release=40)
    low = db_to_gain(-12.0)
    assert np.allclose(env[:80], 1.0) and np.allclose(env[100:200], low)
    assert np.allclose(env[240:], 1.0)
    assert np.all(np.diff(env[80:100]) < 0)  # attack: falling
    assert np.all(np.diff(env[200:240]) > 0)  # release: rising
    assert low < env[80:100].min() and env[99] < env[80]


def test_short_gaps_between_sentences_stay_ducked() -> None:
    env = duck_envelope(400, [(100, 150), (180, 250)], -10.0, attack=20, release=40)
    assert np.allclose(env[100:250], db_to_gain(-10.0))  # 30-sample gap < attack + release


def test_ducking_at_the_edges_of_the_programme() -> None:
    env = duck_envelope(100, [(0, 30), (90, 120)], -6.0, attack=20, release=40)
    assert np.isclose(env[0], db_to_gain(-6.0)) and np.isclose(env[99], db_to_gain(-6.0))
    assert len(env) == 100
    assert np.allclose(duck_envelope(50, [], -6.0, 10, 10), 1.0)
    assert np.allclose(duck_envelope(50, [(10, 20)], 0.0, 10, 10), 1.0)


@given(
    spans=st.lists(st.tuples(st.integers(-50, 600), st.integers(0, 200)), max_size=6),
    depth=st.floats(-30, -1),
    attack=st.integers(0, 80),
    release=st.integers(0, 80),
)
def test_duck_envelope_properties(
    spans: list[tuple[int, int]], depth: float, attack: int, release: int
) -> None:
    total = 500
    sp = [(a, a + n) for a, n in spans]
    env = duck_envelope(total, sp, depth, attack, release)
    low = db_to_gain(depth)
    assert env.shape == (total,)
    assert np.all(env <= 1.0 + 1e-6) and np.all(env >= low - 1e-6)
    for a, b in sp:  # fully down wherever the voice sounds
        lo, hi = max(0, a), min(total, b)
        if lo < hi:
            assert np.allclose(env[lo:hi], low, rtol=1e-5)
    # never a step bigger than one ramp step (no clicks from the envelope itself)
    if attack and release:
        step_db = depth / min(attack, release)
        assert np.abs(np.diff(env)).max() <= 1 - db_to_gain(step_db) + 1e-5


# ---- placing and summing ----------------------------------------------------------------------
def test_pieces_land_on_exact_samples_with_their_gain() -> None:
    out, rep = mix(200, [Layer([Placement(const(50), 100, -6.0)])], S)
    assert out.shape == (200, 1) and out.dtype == np.float32
    assert np.all(out[:100] == 0) and np.all(out[150:] == 0)
    assert np.allclose(out[110:140], db_to_gain(-6.0))  # inside the fades
    assert 0 < out[100, 0] < 0.05 and 0 < out[149, 0] < 0.05
    assert rep.clipped == 0 and np.isclose(rep.peak, db_to_gain(-6.0), rtol=1e-4)


def test_pieces_are_cut_by_the_ends_of_the_programme() -> None:
    out, _ = mix(100, [Layer([Placement(const(50), -20), Placement(const(50), 80)])], S)
    assert out.shape == (100, 1)
    assert np.allclose(out[:20], 1.0) and np.allclose(out[90:], 1.0)  # cut fades are not played
    out, _ = mix(100, [Layer([Placement(const(10), 200), Placement(const(10), -50)])], S)
    assert np.all(out == 0)


def test_channels_are_matched() -> None:
    stereo = MixSettings(sample_rate=RATE, channels=2)
    out, _ = mix(50, [Layer([Placement(np.ones(50, dtype=np.float32), 0)])], stereo)
    assert out.shape == (50, 2) and np.array_equal(out[:, 0], out[:, 1])
    mono_of_stereo, _ = mix(50, [Layer([Placement(const(50, 0.5, ch=2), 0)])], S)
    assert np.isclose(mono_of_stereo[25, 0], 0.5)
    with pytest.raises(ValueError):
        mix(50, [Layer([Placement(const(50, ch=6), 0)])], stereo)


def test_overlaps_add_up_and_clipping_is_reported() -> None:
    out, rep = mix(
        100, [Layer([Placement(const(100, 0.8), 0)]), Layer([Placement(const(100, 0.8), 0)])], S
    )
    assert np.isclose(out[50, 0], 1.6) and rep.clipped > 0 and rep.peak > 1


def test_only_the_layer_that_asks_is_ducked() -> None:
    voice = Layer([Placement(const(100, 0.1), 200)])
    music = Layer([Placement(const(500, 0.5), 0)], duck_db=-20.0, duck_under=[(200, 300)])
    film = Layer([Placement(const(500, 0.2), 0)])
    out, _ = mix(500, [voice, music, film], S)
    assert np.isclose(out[250, 0], 0.1 + 0.5 * db_to_gain(-20.0) + 0.2, rtol=1e-4)
    assert np.isclose(out[50, 0], 0.5 + 0.2, rtol=1e-4)
    assert np.isclose(out[400, 0], 0.5 + 0.2, rtol=1e-4)


def test_cuts_between_pieces_of_the_film_do_not_click() -> None:
    """Two pieces of a 200 Hz tone, cut mid-waveform and butted together at a 20 dB level
    change: with edge fades the seam is no rougher than the tone itself."""
    rate = 48_000
    settings = MixSettings(sample_rate=rate, channels=1)
    a = sine(rate // 2, 200, rate)
    b = sine(rate // 2, 200, rate, phase=2.0)  # starts far from zero
    seam = rate // 2
    layers = [Layer([Placement(a, 0, -20.0), Placement(b, seam, 0.0)])]
    out, _ = mix(rate, layers, settings)
    smooth = 2 * np.pi * 200 / rate  # largest step a full-scale 200 Hz sine makes
    (jump,) = seam_jumps(out, [seam], radius=rate // 100)
    assert jump <= smooth * 1.05

    # The same without fades would click: a jump from ~0.09 to ~0.9 in one sample.
    raw = np.concatenate([a[:, 0] * db_to_gain(-20.0), b[:, 0]])[:, None]
    assert seam_jumps(raw, [seam], radius=rate // 100)[0] > 10 * smooth


def test_seam_jumps_at_the_edges() -> None:
    sig = np.zeros((10, 1), dtype=np.float32)
    sig[0] = 1.0
    assert seam_jumps(sig, [0, 9], radius=2) == [1.0, 0.0]
