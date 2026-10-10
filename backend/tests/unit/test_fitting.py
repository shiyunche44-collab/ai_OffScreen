"""Unit and property tests for duration fitting."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.fitting import (
    MAX_CLIP_MS,
    MAX_SPEED,
    MIN_CLIP_MS,
    MIN_SPEED,
    Clip,
    Shot,
    fit_duration,
)

EPS = 1e-9
FRAME_MS_24 = 1000 / 24


def total_ms(clips: list[Clip]) -> int:
    return sum(c.playback_duration_ms() for c in clips)


def check_limits(clips: list[Clip], shots: list[Shot]) -> None:
    by_id = {s.shot_id: s for s in shots}
    for c in clips:
        shot = by_id[c.shot_id]
        assert MIN_SPEED - EPS <= c.speed <= MAX_SPEED + EPS
        assert c.source_duration_ms <= MAX_CLIP_MS
        assert c.source_duration_ms >= min(MIN_CLIP_MS, shot.available_ms)
        assert shot.start_ms <= c.src_in_ms < c.src_out_ms <= shot.start_ms + shot.available_ms


class TestBasic:
    def test_single_shot_at_normal_speed(self) -> None:
        clips = fit_duration([Shot("sh_001", 3000)], 2000)

        assert len(clips) == 1
        assert clips[0].playback_duration_ms() == 2000
        assert clips[0].speed == pytest.approx(1.0)

    def test_speed_makes_up_the_difference(self) -> None:
        clips = fit_duration([Shot("sh_001", 4000)], 4500)  # 4000 ms of footage, slowed 0.89x

        assert len(clips) == 1
        assert clips[0].playback_duration_ms() == 4500
        assert MIN_SPEED <= clips[0].speed < 1.0

    def test_long_target_uses_several_shots(self) -> None:
        shots = [Shot(f"sh_{i}", 3000, start_ms=i * 10_000) for i in range(6)]

        clips = fit_duration(shots, 12_000)

        assert total_ms(clips) == 12_000
        assert len(clips) >= 3
        check_limits(clips, shots)

    def test_a_segment_length_that_is_not_on_the_frame_grid(self) -> None:
        shots = [Shot(f"sh_{i}", 3000) for i in range(4)]

        assert total_ms(fit_duration(shots, 5230)) == 5230

    def test_prefers_not_slowing_down_when_more_footage_is_available(self) -> None:
        shots = [Shot("sh_a", 4000), Shot("sh_b", 4000)]

        clips = fit_duration(shots, 4700)

        assert len(clips) == 2
        assert all(c.speed >= 1.0 - EPS for c in clips)

    def test_clip_is_cut_from_the_middle_of_the_shot_in_source_time(self) -> None:
        (clip,) = fit_duration([Shot("sh_a", 10_000, start_ms=50_000)], 2000)

        assert (clip.src_in_ms, clip.src_out_ms) == (50_000 + 4000, 50_000 + 6000)

    def test_short_shot_is_used_whole(self) -> None:
        shots = [Shot("sh_short", 500, start_ms=1000), Shot("sh_long", 3000)]

        clips = fit_duration(shots, 3000)

        assert clips[0].shot_id == "sh_short"
        assert (clips[0].src_in_ms, clips[0].src_out_ms) == (1000, 1500)
        assert total_ms(clips) == 3000
        check_limits(clips, shots)

    def test_shots_wrap_around_when_there_are_too_few(self) -> None:
        clips = fit_duration([Shot("sh_a", 2000)], 9000)

        assert total_ms(clips) == 9000
        assert {c.shot_id for c in clips} == {"sh_a"}
        assert len(clips) >= 4

    def test_a_gap_the_first_shot_cannot_cover_is_filled_by_a_longer_one(self) -> None:
        shots = [Shot("sh_short", 800), Shot("sh_long", 4000)]  # 942-1391 ms: not 1 or 2 shorts

        clips = fit_duration(shots, 1000)

        assert total_ms(clips) == 1000
        assert [c.shot_id for c in clips] == ["sh_long"]

    def test_footage_less_shot_is_skipped(self) -> None:
        clips = fit_duration([Shot("sh_zero", 0), Shot("sh_a", 3000)], 2000)

        assert [c.shot_id for c in clips] == ["sh_a"]

    def test_target_shorter_than_any_clip_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot fill"):
            fit_duration([Shot("sh_001", 5000)], 100)

    def test_no_shots_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no shots"):
            fit_duration([], 2000)
        with pytest.raises(ValueError, match="no shots"):
            fit_duration([Shot("sh_zero", 0)], 2000)

    def test_non_positive_target_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            fit_duration([Shot("sh_001", 3000)], 0)


# Shortest playback of a full-size clip: ceil(800 / 1.15). Shots of 1.2 s or more leave no gaps
# between "one clip" and "two clips"; shorter ones can (see the next test).
MIN_TARGET = 696


@given(
    target_ms=st.integers(min_value=MIN_TARGET, max_value=30_000),
    availables=st.lists(st.integers(min_value=1200, max_value=20_000), min_size=1, max_size=6),
)
def test_total_duration_equals_the_target_and_limits_hold(
    target_ms: int, availables: list[int]
) -> None:
    shots = [Shot(f"sh_{i:03d}", a, start_ms=i * 100_000) for i, a in enumerate(availables)]

    clips = fit_duration(shots, target_ms)

    assert abs(total_ms(clips) - target_ms) <= FRAME_MS_24  # in fact exact
    assert total_ms(clips) == target_ms
    check_limits(clips, shots)


@given(
    target_ms=st.integers(min_value=1, max_value=30_000),
    availables=st.lists(st.integers(min_value=1, max_value=20_000), min_size=1, max_size=6),
)
def test_short_shots_and_odd_targets_either_fit_exactly_or_are_rejected(
    target_ms: int, availables: list[int]
) -> None:
    shots = [Shot(f"sh_{i:03d}", a) for i, a in enumerate(availables)]

    try:
        clips = fit_duration(shots, target_ms)
    except ValueError as e:
        assert "cannot fill" in str(e)
        return
    assert total_ms(clips) == target_ms
    check_limits(clips, shots)
