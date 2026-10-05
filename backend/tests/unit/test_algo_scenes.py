from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.scenes import (
    Boundary,
    boundary_features,
    dialogue_gap_ms,
    enforce_min_length,
    merge_segments,
    pick_candidates,
    plan_windows,
    segments_from_cuts,
)
from offscreen.domain.index import TranscriptLine


def line(start: int, end: int) -> TranscriptLine:
    return TranscriptLine(id=f"ln_{start:07d}", start_ms=start, end_ms=end, text="x")


# --- dialogue gaps ----------------------------------------------------------------------


def test_gap_is_zero_while_someone_speaks_across_the_cut() -> None:
    assert dialogue_gap_ms([line(1_000, 5_000)], 3_000, 60_000) == 0


def test_gap_runs_from_the_last_line_before_to_the_first_line_after() -> None:
    lines = [line(0, 2_000), line(10_000, 12_000)]
    assert dialogue_gap_ms(lines, 5_000, 60_000) == 8_000


def test_the_films_ends_bound_the_gap() -> None:
    assert dialogue_gap_ms([line(40_000, 41_000)], 10_000, 60_000) == 40_000  # from 0
    assert dialogue_gap_ms([line(1_000, 2_000)], 30_000, 60_000) == 58_000  # to the end
    assert dialogue_gap_ms([], 30_000, 60_000) == 60_000


def test_a_line_ending_exactly_at_the_cut_does_not_span_it() -> None:
    assert dialogue_gap_ms([line(0, 5_000), line(9_000, 9_500)], 5_000, 60_000) == 4_000


# --- boundary features ------------------------------------------------------------------

RED = [1000] + [0] * 63
BLUE = [0] * 63 + [1000]


def test_features_combine_visual_change_and_dialogue_pauses() -> None:
    ends = [10_000, 20_000, 30_000]
    firsts = [RED, BLUE, BLUE]
    lasts = [RED, BLUE, BLUE]
    lines = [line(14_000, 26_000)]  # speaks across the 2nd cut (20 s)
    b0, b1 = boundary_features(ends, firsts, lasts, lines, 40_000)
    assert (b0.after, b0.time_ms, b0.dissimilarity) == (0, 10_000, 1.0)  # red -> blue
    assert b0.gap_ms == 14_000 - 0 and b0.score == pytest.approx(0.6 * 1.0 + 0.4 * 1.0)
    assert (b1.dissimilarity, b1.gap_ms, b1.score) == (0.0, 0, 0.0)  # same picture, talking


def test_features_reject_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        boundary_features([1, 2], [RED], [RED, RED], [], 10)


# --- candidates -------------------------------------------------------------------------


def b(after: int, t: int, score: float) -> Boundary:
    return Boundary(after, t, score, 0, score)


def test_candidates_need_the_threshold_and_room() -> None:
    bs = [b(0, 10_000, 0.9), b(1, 20_000, 0.8), b(2, 40_000, 0.3), b(3, 60_000, 0.7)]
    # 20 s is within 20 s of the stronger 10 s one -> dropped; 0.3 is below the threshold
    assert pick_candidates(bs, min_gap_ms=20_000, max_span_ms=10**9, duration_ms=90_000) == [0, 3]


def test_the_strongest_wins_a_crowded_stretch() -> None:
    bs = [b(0, 10_000, 0.5), b(1, 15_000, 0.9), b(2, 19_000, 0.6)]
    assert pick_candidates(bs, min_gap_ms=20_000, max_span_ms=10**9, duration_ms=60_000) == [1]


def test_long_stretches_get_their_best_boundary_added() -> None:
    bs = [b(i, (i + 1) * 60_000, 0.1 + 0.01 * (i % 5)) for i in range(20)]  # all weak
    cuts = pick_candidates(
        bs, threshold=0.5, min_gap_ms=20_000, max_span_ms=5 * 60_000, duration_ms=21 * 60_000
    )
    times = [0, *(bs[c].time_ms for c in cuts), 21 * 60_000]
    assert all(z - a <= 5 * 60_000 for a, z in pairwise(times))
    assert cuts == sorted(cuts) and 0 < len(cuts) <= 4


def test_a_long_stretch_with_no_eligible_boundary_is_left_alone() -> None:
    assert pick_candidates([], max_span_ms=1_000, duration_ms=10_000) == []


@given(
    st.lists(
        st.tuples(st.integers(1_000, 3_000_000), st.floats(0, 1)),
        min_size=0,
        max_size=60,
        unique_by=lambda x: x[0],
    ),
    st.integers(1_000, 60_000),
)
def test_candidates_respect_the_minimum_distance(
    items: list[tuple[int, float]], min_gap: int
) -> None:
    bs = [b(i, t, s) for i, (t, s) in enumerate(sorted(items))]
    cuts = pick_candidates(
        bs, threshold=0.0, min_gap_ms=min_gap, max_span_ms=10**9, duration_ms=4_000_000
    )
    times = [bs[c].time_ms for c in cuts]
    assert all(z - a >= min_gap for a, z in pairwise(times))


# --- segments, windows, scenes ----------------------------------------------------------


def test_segments_partition_the_shots() -> None:
    assert segments_from_cuts(10, [2, 6]) == [(0, 3), (3, 7), (7, 10)]
    assert segments_from_cuts(10, []) == [(0, 10)]
    assert segments_from_cuts(10, [9, 9, -1, 2]) == [
        (0, 3),
        (3, 10),
    ]  # out of range / repeats ignored
    assert segments_from_cuts(0, []) == []


@given(st.integers(1, 200), st.lists(st.integers(-5, 205), max_size=40))
def test_segments_always_cover_every_shot_once(n: int, cuts: list[int]) -> None:
    segs = segments_from_cuts(n, cuts)
    assert segs[0][0] == 0 and segs[-1][1] == n
    assert all(hi > lo for lo, hi in segs) and all(a[1] == c[0] for a, c in pairwise(segs))


def test_a_few_segments_fit_one_window() -> None:
    (w,) = plan_windows(5, 8, 2)
    assert (w.lo, w.hi, w.accept_lo, w.accept_hi) == (0, 5, 0, 4)
    assert plan_windows(1, 8, 2) == [] and plan_windows(0, 8, 2) == []


def test_windows_hand_over_in_the_middle_of_their_overlap() -> None:
    ws = plan_windows(20, 8, 2)
    assert [(w.lo, w.hi) for w in ws] == [(0, 8), (6, 14), (12, 20)]
    assert [(w.accept_lo, w.accept_hi) for w in ws] == [(0, 7), (7, 13), (13, 19)]


@given(st.integers(2, 120), st.integers(2, 20), st.data())
def test_every_join_is_decided_exactly_once_inside_its_window(
    n: int, size: int, data: st.DataObject
) -> None:
    overlap = data.draw(st.integers(1, size - 1))
    ws = plan_windows(n, size, overlap)
    decided: list[int] = []
    for w in ws:
        assert w.hi - w.lo <= size and 0 <= w.lo < w.hi <= n
        # a window can only decide joins between two segments it shows
        assert w.lo <= w.accept_lo <= w.accept_hi <= w.hi - 1
        decided += range(w.accept_lo, w.accept_hi)
    assert decided == list(range(n - 1))
    assert ws[-1].hi == n


def test_merging_follows_the_decisions() -> None:
    segs = [(0, 3), (3, 7), (7, 10), (10, 12)]
    assert merge_segments(segs, [True, False, True]) == [(0, 3), (3, 10), (10, 12)]
    assert merge_segments(segs, [False, False, False]) == [(0, 12)]
    assert merge_segments(segs, [True, True, True]) == segs
    assert merge_segments([], []) == []
    with pytest.raises(ValueError):
        merge_segments(segs, [True])


def test_slivers_are_folded_into_a_neighbour() -> None:
    starts = [0, 10_000, 20_000, 22_000, 40_000, 50_000]
    ends = [10_000, 20_000, 22_000, 40_000, 50_000, 60_000]
    spans = [(0, 2), (2, 3), (3, 6)]  # 0-20 s, 20-22 s (a sliver), 22-60 s
    assert enforce_min_length(spans, starts, ends, 10_000) == [(0, 3), (3, 6)]
    assert enforce_min_length([(0, 1), (1, 6)], starts, ends, 20_000) == [
        (0, 6)
    ]  # first folds forward
    assert enforce_min_length([(0, 6)], starts, ends, 10**9) == [(0, 6)]  # a single scene stays


@given(st.lists(st.integers(1, 30), min_size=1, max_size=30), st.integers(1, 60))
def test_min_length_keeps_a_partition_of_the_shots(shot_secs: list[int], min_s: int) -> None:
    starts, t = [], 0
    for s in shot_secs:
        starts.append(t * 1000)
        t += s
    ends = [x + s * 1000 for x, s in zip(starts, shot_secs, strict=True)]
    spans = [(i, i + 1) for i in range(len(shot_secs))]  # every shot its own scene
    out = enforce_min_length(spans, starts, ends, min_s * 1000)
    assert out[0][0] == 0 and out[-1][1] == len(shot_secs)
    assert all(a[1] == c[0] for a, c in pairwise(out))
    if len(out) > 1:
        assert all(ends[hi - 1] - starts[lo] >= min_s * 1000 for lo, hi in out)
