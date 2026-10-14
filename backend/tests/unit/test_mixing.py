from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.mixing import (
    RATE,
    Duck,
    Piece,
    db_to_gain,
    duck_envelope,
    fade_edges,
    loop_to,
    max_step,
    merge_spans,
    mix,
    ms_to_samples,
    peak_db,
)


def tone(n: int, level: float = 0.5) -> np.ndarray:  # type: ignore[type-arg]
    return np.full((n, 2), level, dtype=np.float32)


def test_pieces_are_placed_by_sample_and_summed() -> None:
    out = mix(100, [Piece(10, tone(20, 0.25))], [Piece(20, tone(30, 0.5))])
    assert out.shape == (100, 2) and out.dtype == np.float32
    assert out[5, 0] == 0 and out[15, 0] == 0.25
    assert out[25, 0] == 0.75  # both
    assert out[45, 0] == 0.5 and out[60, 0] == 0


def test_pieces_are_cut_at_the_buffer_edges() -> None:
    out = mix(50, [Piece(-10, tone(20)), Piece(45, tone(20))])
    assert out[0, 0] == 0.5 and out[9, 0] == 0.5 and out[10, 0] == 0
    assert out[44, 0] == 0 and out[49, 0] == 0.5


def test_gain_in_db() -> None:
    out = mix(10, [Piece(0, tone(10, 1.0), gain_db=-6.0206)])
    assert abs(out[0, 0] - 0.5) < 1e-4
    assert db_to_gain(0) == 1.0


def test_fade_edges_ramp_both_ends_and_leave_the_middle() -> None:
    x = fade_edges(tone(100, 1.0), 10)
    assert 0 < x[0, 0] < x[9, 0] < 1.0 and x[50, 0] == 1.0
    assert 0 < x[-1, 0] < x[-10, 0] < 1.0
    short = fade_edges(tone(6, 1.0), 100)  # never more than half each side
    assert short[0, 0] < 1.0 and short.shape == (6, 2)
    assert fade_edges(tone(10), 0) is not None


def test_duck_envelope_reaches_the_depth_at_the_narration_start() -> None:
    duck = Duck(depth_db=-12.0, attack_ms=10, release_ms=20)
    a, r = ms_to_samples(10), ms_to_samples(20)
    env = duck_envelope(10 * RATE // 10, [(RATE // 20, RATE // 10)], duck)
    low = db_to_gain(-12.0)
    lo, hi = RATE // 20, RATE // 10
    assert env[0] == 1.0 and env[lo - a - 1] == 1.0
    assert abs(env[lo] - low) < 1e-5 and abs(env[hi - 1] - low) < 1e-5
    assert np.all(np.diff(env[lo - a : lo + 1]) <= 1e-7)  # monotone down
    assert env[hi + r + 1] == 1.0 if hi + r + 1 < len(env) else True
    assert env.min() >= low - 1e-6 and env.max() <= 1.0


def test_close_spans_keep_the_lower_gain_between_them() -> None:
    duck = Duck(depth_db=-10.0, attack_ms=5, release_ms=5)
    env = duck_envelope(2000, [(500, 700), (720, 900)], duck)
    low = db_to_gain(-10.0)
    assert np.all(env[500:900] <= 1.0) and env[600] == np.float32(low)
    assert env[710] < 1.0  # a short gap never springs back to full level


def test_others_are_ducked_only_where_narration_plays() -> None:
    duck = Duck(depth_db=-6.0206, attack_ms=0, release_ms=0)
    out = mix(300, [Piece(100, tone(100, 0.1))], [Piece(0, tone(300, 1.0), duck=duck)])
    assert out[50, 0] == 1.0 and out[250, 0] == 1.0
    assert abs(out[150, 0] - (0.5 + 0.1)) < 1e-4
    plain = mix(300, [Piece(100, tone(100, 0.1))], [Piece(0, tone(300, 1.0))])
    assert abs(plain[150, 0] - 1.1) < 1e-6


def test_narration_is_never_ducked() -> None:
    duck = Duck(depth_db=-20.0)
    out = mix(100, [Piece(0, tone(100, 0.5), duck=duck)])
    assert out[50, 0] == 0.5


def test_fades_remove_clicks_at_cuts() -> None:
    n = ms_to_samples(200)
    hard = mix(n, [], [Piece(0, tone(n, 0.8))])
    soft = mix(n, [], [Piece(0, tone(n, 0.8), fade_ms=15)])
    assert max_step(hard) == 0.0  # constant: edges are the buffer's own
    assert max_step(soft) < 0.8 / ms_to_samples(15) * 1.5
    cut = mix(2 * n, [], [Piece(0, tone(n, 0.8), fade_ms=15), Piece(n, tone(n, -0.8), fade_ms=15)])
    unfaded = mix(2 * n, [], [Piece(0, tone(n, 0.8)), Piece(n, tone(n, -0.8))])
    assert max_step(cut) < 0.05 and max_step(unfaded) > 1.5


def test_loop_merge_and_peak() -> None:
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert loop_to(x, 7).shape == (7, 2) and loop_to(x, 7)[3, 0] == 0
    assert loop_to(x, 0).shape == (0, 2)
    assert merge_spans([(5, 9), (0, 3), (3, 4), (8, 12), (20, 20)]) == [(0, 4), (5, 12)]
    assert peak_db(tone(4, 1.0)) == 0.0 and peak_db(tone(4, 0.0)) == float("-inf")


spans = st.lists(
    st.tuples(st.integers(0, 3000), st.integers(0, 3000)).map(lambda t: (min(t), max(t))),
    max_size=6,
)


@given(spans, st.floats(-30, 0), st.integers(0, 40), st.integers(0, 40))
def test_envelope_stays_between_depth_and_unity(
    sp: list[tuple[int, int]], depth: float, atk: int, rel: int
) -> None:
    duck = Duck(depth_db=depth, attack_ms=atk, release_ms=rel)
    env = duck_envelope(3000, sp, duck)
    low = db_to_gain(depth)
    assert len(env) == 3000
    assert env.min() >= low - 1e-6 and env.max() <= 1.0 + 1e-6
    for lo, hi in merge_spans(sp):
        assert np.all(env[lo : min(hi, 3000)] <= low + 1e-6)


@given(st.integers(1, 500), st.integers(0, 400), st.integers(0, 200))
def test_mix_length_is_exact(total: int, start: int, length: int) -> None:
    out = mix(total, [Piece(start, tone(length))], [Piece(-5, tone(length), duck=Duck())])
    assert out.shape == (total, 2)
