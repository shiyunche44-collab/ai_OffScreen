from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.shots import MAX_SHOT_MS, MIN_SHOT_MS, normalize_shots


def test_clean_input_is_unchanged() -> None:
    raw = [(0, 3000), (3000, 5000), (5000, 9000)]
    assert normalize_shots(raw, 9000) == raw


def test_no_detections_gives_one_shot() -> None:
    assert normalize_shots([], 4000) == [(0, 4000)]


def test_fragments_are_merged_and_long_shot_is_split_evenly() -> None:
    raw = [(0, 100), (100, 300), (300, 20300), (20300, 20400)]
    assert normalize_shots(raw, 20400) == [
        (0, 6800),
        (6800, 13600),
        (13600, 20400),
    ]


def test_first_and_last_short_shots_are_absorbed() -> None:
    assert normalize_shots([(0, 200), (200, 4000), (4000, 4100)], 4100) == [(0, 4100)]
    assert normalize_shots([(0, 200), (200, 1000), (1000, 3000)], 3000) == [(0, 1000), (1000, 3000)]


def test_gaps_and_out_of_range_cuts_are_repaired() -> None:
    # Detector skipped 1000-2000, reported a cut past the end and one at 0.
    assert normalize_shots([(100, 1000), (2000, 3000), (9999, 12000)], 5000) == [
        (0, 2000),
        (2000, 5000),
    ]


def test_video_shorter_than_min_is_kept_whole() -> None:
    assert normalize_shots([(0, 100), (100, 300)], 300) == [(0, 300)]


def test_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        normalize_shots([], 0)
    with pytest.raises(ValueError):
        normalize_shots([], 1000, min_ms=5000, max_ms=8000)


@given(
    cuts=st.lists(st.integers(-1000, 400_000), max_size=60),
    duration=st.integers(1, 300_000),
)
def test_result_is_a_partition_within_bounds(cuts: list[int], duration: int) -> None:
    starts = [0, *sorted(cuts)]
    raw = [(a, b) for a, b in zip(starts, [*starts[1:], duration], strict=True) if b > a]
    out = normalize_shots(raw, duration)

    assert out[0][0] == 0 and out[-1][1] == duration
    assert all(a[1] == b[0] for a, b in pairwise(out))
    assert all(0 < e - s <= MAX_SHOT_MS for s, e in out)
    if duration >= MIN_SHOT_MS:
        assert all(e - s >= MIN_SHOT_MS for s, e in out)
    else:
        assert out == [(0, duration)]


@given(
    cuts=st.lists(st.integers(1, 100_000), max_size=40),
    duration=st.integers(1000, 100_000),
)
def test_normalizing_twice_changes_nothing(cuts: list[int], duration: int) -> None:
    starts = [0, *sorted(set(cuts))]
    raw = [(a, b) for a, b in zip(starts, [*starts[1:], duration], strict=False) if b > a]
    once = normalize_shots(raw, duration)
    assert normalize_shots(once, duration) == once
