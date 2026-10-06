"""Scoring detected shot cuts against hand-marked ones."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from offscreen.algo.cuts import evaluate_cuts, shot_cut_frames
from offscreen.domain.common import Rational
from offscreen.domain.index import CutAnnotations
from offscreen.store.annotations import AnnotationsStore


def test_perfect_and_empty_scores() -> None:
    s = evaluate_cuts([10, 50, 90], [10, 50, 90])
    assert (s.precision, s.recall, s.f1, s.true_positives) == (1.0, 1.0, 1.0, 3)
    assert s.false_positives == () and s.false_negatives == ()
    empty = evaluate_cuts([], [])
    assert (empty.precision, empty.recall, empty.f1) == (1.0, 1.0, 1.0)
    assert evaluate_cuts([5], []).precision == 0.0 and evaluate_cuts([5], []).f1 == 0.0
    assert evaluate_cuts([], [5]).recall == 0.0


def test_tolerance_is_inclusive_and_misses_are_listed() -> None:
    s = evaluate_cuts([12, 60, 200], [10, 50, 100], tolerance=2)
    assert s.matches == ((12, 10),)
    assert s.false_positives == (60, 200) and s.false_negatives == (50, 100)
    assert s.precision == pytest.approx(1 / 3) and s.recall == pytest.approx(1 / 3)
    assert evaluate_cuts([13], [10], tolerance=2).true_positives == 0
    assert evaluate_cuts([13], [10], tolerance=3).true_positives == 1
    assert evaluate_cuts([10], [10], tolerance=0).true_positives == 1


def test_a_cut_found_twice_counts_once() -> None:
    s = evaluate_cuts([10, 11], [10])
    assert s.true_positives == 1 and s.false_positives == (11,)
    assert s.matches == ((10, 10),)
    # the closer one wins even when it comes second
    t = evaluate_cuts([9, 10], [10])
    assert t.matches == ((10, 10),) and t.false_positives == (9,)


def test_one_detection_cannot_serve_two_marks() -> None:
    s = evaluate_cuts([10], [9, 11])
    assert s.true_positives == 1 and len(s.false_negatives) == 1


def test_a_negative_tolerance_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        evaluate_cuts([1], [1], tolerance=-1)


@given(
    st.lists(st.integers(1, 300), unique=True, max_size=30).map(sorted),
    st.lists(st.integers(1, 300), unique=True, max_size=30).map(sorted),
    st.integers(0, 5),
)
def test_scores_are_consistent(detected: list[int], marked: list[int], tol: int) -> None:
    s = evaluate_cuts(detected, marked, tol)
    assert s.true_positives + len(s.false_positives) == len(detected)
    assert s.true_positives + len(s.false_negatives) == len(marked)
    assert 0.0 <= s.f1 <= 1.0 and 0.0 <= s.precision <= 1.0 and 0.0 <= s.recall <= 1.0
    assert all(abs(d - m) <= tol for d, m in s.matches)
    assert len({d for d, _ in s.matches}) == len({m for _, m in s.matches}) == s.true_positives
    # exact cuts always score perfectly
    perfect = evaluate_cuts(marked, marked, tol)
    assert perfect.f1 == 1.0 and perfect.true_positives == len(marked)


def test_shot_starts_become_cut_frames_at_the_movies_frame_rate() -> None:
    fps = Rational(num=24000, den=1001)
    # 0 ms is no cut; 1001 ms is frame 24; a start in the same frame as another is one cut
    assert shot_cut_frames([0, 1001, 1002, 2002], fps) == [24, 48]
    assert shot_cut_frames([], fps) == [] and shot_cut_frames([0], fps) == []


# --- the document and its store -------------------------------------------------------------


def test_annotations_must_be_ascending_and_start_after_frame_zero() -> None:
    fps = Rational(num=24, den=1)
    CutAnnotations(asset_id="ast_x", fps=fps, cuts=[1, 5, 9])
    for bad in ([0, 3], [5, 3], [4, 4], [-1]):
        with pytest.raises(ValidationError):
            CutAnnotations(asset_id="ast_x", fps=fps, cuts=bad)


def test_the_store_round_trips_and_refuses_odd_ids(tmp_path: Path) -> None:
    store = AnnotationsStore(tmp_path)
    assert store.read_cuts("ast_x") is None
    doc = CutAnnotations(asset_id="ast_x", fps=Rational(num=24, den=1), cuts=[10, 20])
    store.write_cuts(doc)
    assert store.read_cuts("ast_x") == doc
    assert (tmp_path / "annotations" / "ast_x" / "cuts.json").is_file()
    for bad in ("", "..", "a/b"):
        with pytest.raises(ValueError, match="bad asset id"):
            store.read_cuts(bad)
