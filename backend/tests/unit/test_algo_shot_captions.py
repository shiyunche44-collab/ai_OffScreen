from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.shot_captions import (
    IMAGE_TOKENS,
    batches,
    dialogue_near,
    estimate_tokens,
    missing,
)
from offscreen.domain.index import TranscriptLine


def line(i: int, start: int, end: int, text: str) -> TranscriptLine:
    return TranscriptLine(id=f"ln_{i:04d}", start_ms=start, end_ms=end, text=text)


LINES = [
    line(1, 0, 1000, "far before"),
    line(2, 9_000, 10_500, "just before"),
    line(3, 11_000, 12_000, "inside"),
    line(4, 13_800, 14_500, "just after"),
    line(5, 30_000, 31_000, "far after"),
]


def test_dialogue_near_includes_lines_within_the_padding() -> None:
    assert dialogue_near(LINES, 11_000, 14_000) == "just before inside just after"
    assert dialogue_near(LINES, 11_000, 14_000, pad_ms=0) == "inside just after"  # starts at 13.8 s
    assert dialogue_near(LINES, 20_000, 21_000) == ""


def test_dialogue_near_cuts_long_text() -> None:
    long = [line(1, 0, 1000, "字" * 500)]
    out = dialogue_near(long, 0, 1000, max_chars=50)
    assert len(out) == 50 and out.endswith("…")


def test_dialogue_near_skips_blank_lines() -> None:
    assert dialogue_near([line(1, 0, 1000, "  ")], 0, 1000) == ""


@given(st.lists(st.integers(), max_size=60), st.integers(1, 12))
def test_batches_cover_every_item_once_in_order(items: list[int], size: int) -> None:
    out = batches(items, size)
    assert [x for b in out for x in b] == items
    assert all(1 <= len(b) <= size for b in out)
    assert all(len(b) == size for b in out[:-1])


def test_batches_rejects_a_bad_size() -> None:
    with pytest.raises(ValueError):
        batches([1], 0)


def test_estimate_grows_with_shots_frames_and_dialogue() -> None:
    base = estimate_tokens(100, batch_size=8, frames_per_shot=3, dialogue_chars=5_000)
    assert base > 100 * 3 * IMAGE_TOKENS  # images dominate
    assert estimate_tokens(200, batch_size=8, frames_per_shot=3, dialogue_chars=5_000) > base
    assert estimate_tokens(100, batch_size=8, frames_per_shot=2, dialogue_chars=5_000) < base
    assert estimate_tokens(100, batch_size=8, frames_per_shot=3, dialogue_chars=9_000) > base
    assert (
        estimate_tokens(100, batch_size=2, frames_per_shot=3, dialogue_chars=5_000) > base
    )  # more requests
    assert estimate_tokens(0, batch_size=8, frames_per_shot=3, dialogue_chars=0) == 0


def test_missing_ids_keep_request_order() -> None:
    assert missing(["a", "b", "c", "d"], ["c", "a", "zzz"]) == ["b", "d"]
    assert missing(["a"], ["a"]) == []
