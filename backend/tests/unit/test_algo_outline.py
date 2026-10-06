"""Outline time allocation and rule checks."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.outline import MIN_BEAT_S, BeatDraft, check_outline, fit_durations


@given(
    weights=st.lists(st.floats(min_value=0.01, max_value=1000), min_size=1, max_size=12),
    extra=st.integers(min_value=0, max_value=900),
)
def test_fitted_seconds_add_up_exactly_and_respect_the_minimum(
    weights: list[float], extra: int
) -> None:
    total = MIN_BEAT_S * len(weights) + extra
    parts = fit_durations(weights, total)
    assert sum(parts) == total
    assert all(p >= MIN_BEAT_S for p in parts)
    assert all(isinstance(p, int) for p in parts)


@given(
    weights=st.lists(st.floats(min_value=0.01, max_value=1000), min_size=2, max_size=8),
    extra=st.integers(min_value=0, max_value=600),
)
def test_a_heavier_beat_never_gets_less_time(weights: list[float], extra: int) -> None:
    parts = fit_durations(weights, MIN_BEAT_S * len(weights) + extra)
    for i, a in enumerate(weights):
        for j, b in enumerate(weights):
            if a > b:
                assert parts[i] >= parts[j]


def test_examples() -> None:
    assert fit_durations([1, 1], 10) == [5, 5]
    assert fit_durations([3, 1], 12) == [9, 3]
    assert fit_durations([1, 100], 20) == [3, 17]  # the light beat is held at the minimum
    assert fit_durations([15, 45], 60) == [15, 45]  # proportions survive when nothing is short
    assert fit_durations([1], 7) == [7]
    assert fit_durations([10, 10, 10], 10, min_s=1) == [4, 3, 3]  # remainder to the earlier one


def test_fit_rejects_impossible_requests() -> None:
    with pytest.raises(ValueError, match="at least"):
        fit_durations([1, 1, 1], 8)
    with pytest.raises(ValueError):
        fit_durations([], 10)
    with pytest.raises(ValueError):
        fit_durations([1, 0], 10)


SCENES = ["sc_001", "sc_002", "sc_003"]


def test_a_good_outline_has_no_problems() -> None:
    beats = [BeatDraft("hook", ["sc_001"], 20), BeatDraft("ending", ["sc_002", "sc_003"], 40)]
    assert check_outline(beats, SCENES, target_s=60) == []


def test_problems_are_reported_in_words_for_the_model() -> None:
    beats = [
        BeatDraft("hook", ["sc_404"], 2),
        BeatDraft("hook", [], 10),
    ]
    problems = check_outline(beats, SCENES, target_s=60)
    joined = "\n".join(problems)
    assert "sc_404" in joined  # unknown scene
    assert "重复" in joined  # duplicate beat name
    assert "没有 scene_refs" in joined
    assert "至少 3 秒" in joined
    assert "合计 12 秒" in joined and "目标 60 秒" in joined


def test_the_total_may_stray_within_the_tolerance() -> None:
    beats = [BeatDraft("a", ["sc_001"], 30), BeatDraft("b", ["sc_002"], 36)]
    assert check_outline(beats, SCENES, target_s=60) == []  # 66 s is within 20 %
    assert check_outline(beats, SCENES, target_s=50)  # 66 s is not
