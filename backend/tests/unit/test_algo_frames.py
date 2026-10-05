from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.frames import (
    SHARPNESS_SCALE,
    brightness_score,
    color_histogram,
    frame_quality,
    frame_times,
    histogram_similarity,
    laplacian_variance,
    sharpness_score,
    shot_quality,
    sprite_slot,
)

# --- sampling ---------------------------------------------------------------------------


def test_frame_times_at_ten_fifty_ninety_percent() -> None:
    assert frame_times(1000, 2000) == [1100, 1500, 1900]
    assert frame_times(0, 8000) == [800, 4000, 7200]


def test_frame_times_rejects_an_empty_range() -> None:
    with pytest.raises(ValueError):
        frame_times(5, 5)


@given(start=st.integers(0, 10**7), length=st.integers(1, 10**6))
def test_frame_times_stay_inside_the_shot_and_in_order(start: int, length: int) -> None:
    end = start + length
    times = frame_times(start, end)
    assert len(times) == 3
    assert all(start <= t < end for t in times)
    assert times == sorted(times)


@given(start=st.integers(0, 10**6), length=st.integers(500, 8000))
def test_frame_times_of_normal_shots_are_distinct(start: int, length: int) -> None:
    assert len(set(frame_times(start, start + length))) == 3  # shots are >= 0.5 s


# --- metrics ----------------------------------------------------------------------------


def checker(size: int = 64, cell: int = 1) -> np.ndarray:
    y, x = np.indices((size, size))
    return (((x // cell + y // cell) % 2) * 255).astype(np.uint8)


def test_flat_frames_have_no_detail_and_black_is_dark() -> None:
    black = np.zeros((48, 64), dtype=np.uint8)
    assert laplacian_variance(black) == 0.0
    assert frame_quality(black) == (0.0, 0.0)
    white = np.full((48, 64), 255, dtype=np.uint8)
    assert frame_quality(white) == (0.0, 1.0)  # bright but featureless


def test_detail_scores_higher_than_blur() -> None:
    sharp = checker(64, cell=1)
    soft = checker(64, cell=8)  # same pattern, coarse: few edges
    assert laplacian_variance(sharp) > laplacian_variance(soft) > 0
    assert frame_quality(sharp)[0] > frame_quality(soft)[0]


def test_a_blurred_copy_scores_lower_than_the_original() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(90, 160), dtype=np.uint8)
    f = img.astype(np.float64)
    blurred = (
        (f + np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5
    ).astype(np.uint8)
    assert frame_quality(blurred)[0] < frame_quality(img)[0]


def test_brightness_is_mean_luma() -> None:
    assert brightness_score(np.full((4, 4), 51, dtype=np.uint8)) == pytest.approx(0.2)
    assert brightness_score(np.zeros((0, 0), dtype=np.uint8)) == 0.0


def test_tiny_images_are_handled() -> None:
    assert laplacian_variance(np.zeros((2, 2), dtype=np.uint8)) == 0.0
    assert laplacian_variance(np.zeros((5,), dtype=np.uint8)) == 0.0


def test_sharpness_score_is_monotone_and_bounded() -> None:
    assert sharpness_score(0) == 0.0 and sharpness_score(-5) == 0.0
    assert sharpness_score(SHARPNESS_SCALE) == pytest.approx(0.5)
    assert sharpness_score(1e12) < 1.0


@given(st.floats(0, 1e9), st.floats(0, 1e9))
def test_sharpness_score_never_decreases(a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert 0.0 <= sharpness_score(lo) <= sharpness_score(hi) <= 1.0


@given(st.integers(0, 255), st.integers(1, 12), st.integers(1, 12))
def test_brightness_of_a_flat_frame_is_its_level(level: int, h: int, w: int) -> None:
    assert brightness_score(np.full((h, w), level, dtype=np.uint8)) == pytest.approx(level / 255)


def test_shot_quality_is_the_mean_of_its_frames() -> None:
    q = shot_quality([(0.2, 0.4), (0.4, 0.6), (0.6, 0.8)])
    assert (q.sharpness, q.brightness) == (pytest.approx(0.4), pytest.approx(0.6))
    with pytest.raises(ValueError):
        shot_quality([])


# --- sprite layout ----------------------------------------------------------------------


def test_sprite_slots_fill_rows_then_sheets() -> None:
    assert sprite_slot(0, 10, 10) == (0, 0, 0)
    assert sprite_slot(9, 10, 10) == (0, 9, 0)
    assert sprite_slot(10, 10, 10) == (0, 0, 1)
    assert sprite_slot(99, 10, 10) == (0, 9, 9)
    assert sprite_slot(100, 10, 10) == (1, 0, 0)
    with pytest.raises(ValueError):
        sprite_slot(-1, 10, 10)


@given(st.integers(0, 10**5), st.integers(1, 20), st.integers(1, 20))
def test_sprite_slots_are_unique_and_inside_the_sheet(n: int, cols: int, rows: int) -> None:
    sheet, col, row = sprite_slot(n, cols, rows)
    assert 0 <= col < cols and 0 <= row < rows
    assert sheet * cols * rows + row * cols + col == n  # a bijection, so no two shots collide


# --- colour histograms ------------------------------------------------------------------


def solid(r: int, g: int, b: int, h: int = 9, w: int = 16) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[...] = (r, g, b)
    return img


def test_a_solid_colour_fills_one_bin() -> None:
    hist = color_histogram(solid(255, 0, 0))
    assert sum(hist) == 1000 and len(hist) == 64
    assert hist[3 * 16 + 0 * 4 + 0] == 1000  # r=3, g=0, b=0
    assert sum(1 for v in hist if v) == 1


def test_histogram_sums_to_1000_whatever_the_pixel_count() -> None:
    rng = np.random.default_rng(7)
    for h, w in [(1, 1), (3, 7), (36, 64), (5, 11)]:
        img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        assert sum(color_histogram(img)) == 1000


def test_similarity_is_one_for_the_same_picture_and_zero_for_disjoint_colours() -> None:
    red, blue = color_histogram(solid(255, 0, 0)), color_histogram(solid(0, 0, 255))
    assert histogram_similarity(red, red) == 1.0
    assert histogram_similarity(red, blue) == 0.0
    half = color_histogram(np.concatenate([solid(255, 0, 0, 9, 8), solid(0, 0, 255, 9, 8)], axis=1))
    assert histogram_similarity(half, red) == pytest.approx(0.5)


def test_similar_lighting_scores_higher_than_a_different_place() -> None:
    rng = np.random.default_rng(3)
    day = rng.integers(60, 220, size=(36, 64, 3)).astype(np.uint8)  # a varied picture
    dimmer = (day.astype(np.float64) * 0.92).astype(np.uint8)  # same place, slightly darker
    night = rng.integers(0, 70, size=(36, 64, 3)).astype(np.uint8)  # a different place
    base = color_histogram(day)
    assert histogram_similarity(base, color_histogram(dimmer)) > 0.6
    assert (
        histogram_similarity(base, color_histogram(dimmer))
        > histogram_similarity(base, color_histogram(night)) + 0.3
    )


def test_histogram_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        color_histogram(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        histogram_similarity([1, 2], [1])


@given(
    st.integers(0, 255),
    st.integers(0, 255),
    st.integers(0, 255),
    st.integers(0, 255),
    st.integers(0, 255),
    st.integers(0, 255),
)
def test_similarity_is_symmetric_and_bounded(
    r1: int, g1: int, b1: int, r2: int, g2: int, b2: int
) -> None:
    a, b = color_histogram(solid(r1, g1, b1)), color_histogram(solid(r2, g2, b2))
    assert histogram_similarity(a, b) == histogram_similarity(b, a)
    assert 0.0 <= histogram_similarity(a, b) <= 1.0
