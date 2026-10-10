"""The selection metrics, on rankings small enough to count by hand."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.selection_eval import evaluate_selection

RANKINGS = {
    "a": ["s1", "s2", "s3", "s4", "s5", "s6"],  # first choice right
    "b": ["s9", "s2", "s3", "s4", "s5", "s6"],  # right one second
    "c": ["s9", "s8", "s7", "s6", "s5", "s1"],  # right one only sixth: outside the top 5
    "d": [],  # nothing ranked
}
GOOD = {"a": {"s1"}, "b": {"s2", "s3"}, "c": {"s1"}, "d": {"s1"}}


def test_the_four_numbers() -> None:
    score = evaluate_selection(RANKINGS, GOOD, k=5)

    by_id = {r.label_id: r for r in score.labels}
    assert [by_id[x].first_rank for x in "abcd"] == [1, 2, 6, None]
    assert score.first_choice_rate == 0.25
    assert score.top_k_hit_rate == 0.5  # a and b
    # a: 1/1, b: both acceptable shots are in the top 5 (2/2), c and d: 0
    assert score.top_k_recall == pytest.approx((1 + 1 + 0 + 0) / 4)
    assert score.mean_reciprocal_rank == pytest.approx((1 + 1 / 2 + 1 / 6 + 0) / 4)
    assert by_id["d"].first_choice is None and by_id["a"].first_choice == "s1"


def test_recall_is_capped_by_how_many_could_fit() -> None:
    many = {"x": {f"g{i}" for i in range(10)}}

    full = evaluate_selection({"x": [f"g{i}" for i in range(10)]}, many, k=3)
    part = evaluate_selection({"x": ["g0", "n1", "n2", "g1"]}, many, k=3)

    assert full.top_k_recall == 1.0  # 3 of the 3 places a label of ten could fill
    assert part.top_k_recall == pytest.approx(1 / 3)


def test_k_changes_what_counts_as_found() -> None:
    assert evaluate_selection(RANKINGS, GOOD, k=6).top_k_hit_rate == 0.75
    assert evaluate_selection(RANKINGS, GOOD, k=1).top_k_hit_rate == 0.25


def test_a_label_without_a_ranking_is_a_miss_and_bad_input_is_refused() -> None:
    assert evaluate_selection({}, {"a": {"s1"}}).first_choice_rate == 0.0
    with pytest.raises(ValueError, match="no labels"):
        evaluate_selection({}, {})
    with pytest.raises(ValueError, match="k must"):
        evaluate_selection({}, {"a": {"s1"}}, k=0)
    with pytest.raises(ValueError, match="no acceptable"):
        evaluate_selection({}, {"a": set()})


@given(
    ranking=st.lists(st.integers(0, 20), unique=True, max_size=20),
    good=st.sets(st.integers(0, 20), min_size=1, max_size=8),
    k=st.integers(1, 10),
)
def test_rates_are_ordered_and_within_bounds(ranking: list[int], good: set[int], k: int) -> None:
    score = evaluate_selection({"x": [str(i) for i in ranking]}, {"x": {str(i) for i in good}}, k)

    assert 0 <= score.first_choice_rate <= score.top_k_hit_rate <= 1
    assert 0 <= score.top_k_recall <= score.top_k_hit_rate  # hits are what recall is made of
    assert 0 <= score.mean_reciprocal_rank <= 1
    assert score.first_choice_rate <= score.mean_reciprocal_rank
