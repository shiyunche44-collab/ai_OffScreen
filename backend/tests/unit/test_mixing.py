from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.mixing import (
    MusicPlaced,
    Placed,
    adjacent_jump,
    db_to_gain,
    duck_envelope,
    mix,
    ms_to_samples,
    with_fades,
)

RATE = 1000  # 1 sample per ms keeps the arithmetic readable


def const(n: int, v: float = 1.0) -> np.ndarray:
    return np.full((n, 2), v, dtype=np.float32)


def test_db_to_gain() -> None:
    assert db_to_gain(0) == 1.0
    assert abs(db_to_gain(-6.0206) - 0.5) < 1e-4
    assert abs(db_to_gain(20) - 10.0) < 1e-9


def test_fades_ramp_both_ends_and_leave_the_middle() -> None:
    x = with_fades(const(100), 10)
    assert x[0, 0] < 0.1 and x[-1, 0] < 0.1
    assert np.all(np.diff(x[:10, 0]) > 0) and np.all(np.diff(x[-10:, 0]) < 0)
    assert np.all(x[10:90] == 1.0)


def test_fades_never_overlap_on_a_short_piece() -> None:
    x = with_fades(const(6), 10)
    assert x.max() <= 1.0 and len(x) == 6
    assert len(with_fades(const(0), 10)) == 0


def test_mix_places_by_sample_and_pads_to_total() -> None:
    out = mix(100, rate=RATE, narration=[Placed(const(20), 30, 0.0, fade_ms=0)])
    assert out.shape == (100, 2)
    assert np.all(out[:30] == 0) and np.all(out[30:50] == 1.0) and np.all(out[50:] == 0)


def test_mix_cuts_pieces_at_both_edges_of_the_buffer() -> None:
    out = mix(
        50,
        rate=RATE,
        source=[Placed(const(30), -10, 0.0, fade_ms=0), Placed(const(30), 40, 0.0, fade_ms=0)],
    )
    assert np.all(out[:20] == 1.0) and np.all(out[20:40] == 0) and np.all(out[40:] == 1.0)


def test_mix_applies_gain_and_sums_without_normalizing() -> None:
    out = mix(
        10,
        rate=RATE,
        narration=[Placed(const(10, 0.5), 0, fade_ms=0)],
        source=[Placed(const(10, 0.5), 0, -6.0206, fade_ms=0)],
    )
    assert np.allclose(out, 0.75, atol=1e-3)


def test_duck_envelope_dips_before_and_recovers_after() -> None:
    env = duck_envelope(
        2000, [(800, 1200)], depth_db=-12.0, rate=RATE, attack_ms=100, release_ms=300
    )
    floor = db_to_gain(-12.0)
    assert env[0] == 1.0 and env[-1] == 1.0
    assert env[700] == 1.0 and env[799] < 1.0  # starts falling 100 ms ahead
    assert np.allclose(env[800:1200], floor, atol=1e-6)
    assert env[1200] == np.float32(floor)
    assert env[1500] == 1.0 and 0 < env[1350] < 1.0
    assert np.all(np.diff(env[700:800]) <= 0) and np.all(np.diff(env[1200:1500]) >= 0)


def test_duck_envelope_stays_down_through_a_short_pause() -> None:
    env = duck_envelope(
        3000, [(500, 1000), (1100, 1600)], depth_db=-10.0, rate=RATE, attack_ms=50, release_ms=400
    )
    # The gap is shorter than the release: the gain only recovers part of the way.
    assert env[1050] < db_to_gain(-10.0) * 1.5 and env[1050] < 0.5
    assert env[1600 + 400] == 1.0


def test_music_is_ducked_under_narration_only() -> None:
    out = mix(
        2000,
        rate=RATE,
        narration=[Placed(const(400, 0.0), 800, fade_ms=0)],
        music=[MusicPlaced(const(2000), 0, 2000, gain_db=0.0, duck_depth_db=-20.0, fade_ms=0)],
        attack_ms=50,
        release_ms=100,
    )
    assert out[100, 0] == 1.0 and out[1900, 0] == 1.0
    assert abs(out[1000, 0] - db_to_gain(-20.0)) < 1e-5


def test_music_without_duck_and_cut_at_its_end() -> None:
    out = mix(
        100,
        rate=RATE,
        narration=[Placed(const(50, 0.0), 10, fade_ms=0)],
        music=[MusicPlaced(const(500, 0.5), 20, 60, gain_db=0.0, fade_ms=0)],
    )
    assert np.all(out[:20] == 0) and np.all(out[20:60] == 0.5) and np.all(out[60:] == 0)


def test_fades_remove_the_click_between_cut_pieces() -> None:
    hard = mix(
        400,
        rate=RATE,
        source=[Placed(const(200, 0.8), 0, fade_ms=0), Placed(const(200, -0.8), 200, fade_ms=0)],
    )
    soft = mix(
        400,
        rate=RATE,
        source=[Placed(const(200, 0.8), 0, fade_ms=20), Placed(const(200, -0.8), 200, fade_ms=20)],
    )
    assert adjacent_jump(hard) == np.float32(1.6)
    assert adjacent_jump(soft) < 0.1


def test_ms_to_samples() -> None:
    assert ms_to_samples(1000, 48000) == 48000 and ms_to_samples(10, 48000) == 480


@given(
    st.integers(1, 300),
    st.lists(st.tuples(st.integers(-50, 300), st.integers(1, 120)), max_size=6),
    st.floats(-30, 0),
)
def test_mix_length_and_duck_range_hold_for_any_layout(
    total: int, pieces: list[tuple[int, int]], depth: float
) -> None:
    narr = [Placed(const(n, 0.1), s) for s, n in pieces]
    out = mix(
        total,
        rate=RATE,
        narration=narr,
        music=[MusicPlaced(const(total, 0.2), 0, total, duck_depth_db=depth)],
    )
    assert out.shape == (total, 2) and np.isfinite(out).all()
    env = duck_envelope(total, [(s, s + n) for s, n in pieces], depth_db=depth, rate=RATE)
    assert env.min() >= db_to_gain(depth) - 1e-6 and env.max() <= 1.0


# ---- music: fit, loop, fade out -----------------------------------------------------------
def ramp(n: int) -> np.ndarray:
    return np.stack([np.arange(n, dtype=np.float32)] * 2, axis=1)


def test_fit_music_trims_a_long_track_and_fades_out() -> None:
    from offscreen.algo.mixing import fit_music

    out = fit_music(const(1000), 400, fade_out_n=100)
    assert len(out) == 400 and np.all(out[:300] == 1.0)
    assert out[300, 0] == 1.0 and out[-1, 0] == 0.0 and np.all(np.diff(out[300:, 0]) <= 0)


def test_fit_music_loops_a_short_track_to_the_exact_length() -> None:
    from offscreen.algo.mixing import fit_music

    for length in (250, 301, 1000):
        assert len(fit_music(const(100), length)) == length
        assert len(fit_music(const(100), length, crossfade_n=20)) == length
    assert fit_music(const(100), 0).shape == (0, 2)
    assert np.all(fit_music(np.zeros((0, 2), dtype=np.float32), 50) == 0)


def test_loop_seams_are_crossfaded_without_a_jump() -> None:
    from offscreen.algo.mixing import fit_music

    x = np.sin(np.linspace(0, 6.0, 300, dtype=np.float32))[:, None].repeat(2, axis=1)
    hard = fit_music(x, 900)
    soft = fit_music(x, 900, crossfade_n=60)
    assert adjacent_jump(hard) > 0.2  # the sine does not end where it began
    assert adjacent_jump(soft) < 0.05
    assert np.allclose(soft[: 300 - 60], x[: 300 - 60])  # the first pass is untouched


def test_music_in_the_mix_loops_and_fades_at_its_end() -> None:
    out = mix(
        1000,
        rate=RATE,
        music=[
            MusicPlaced(
                const(300), 100, 900, gain_db=0.0, fade_ms=0, fade_out_ms=200, crossfade_ms=0
            )
        ],
    )
    assert np.all(out[:100] == 0) and np.all(out[100:700] == 1.0) and np.all(out[900:] == 0)
    assert out[899, 0] < 0.02 and out[800, 0] < 0.6
