from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.matching import MAX_CLIP_MS, MIN_CLIP_MS, Span, fill_clips, order_candidates


def spans(*durs: int, gap: int = 100) -> list[Span]:
    out, t = [], 0
    for i, d in enumerate(durs):
        out.append(Span(f"sh_{i}", t, t + d))
        t += d + gap
    return out


def test_span_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Span("sh_0", 5, 5)


def test_order_candidates_unused_first_then_chronological() -> None:
    shots = list(reversed(spans(1000, 1000, 1000, 1000)))
    got = [s.shot_id for s in order_candidates(shots, {"sh_0", "sh_2"})]
    assert got == ["sh_1", "sh_3", "sh_0", "sh_2"]


def test_exact_fill_with_middle_crop_of_long_shots() -> None:
    (c,) = fill_clips(spans(10_000), 3_000)
    assert (c.src_in_ms, c.src_out_ms) == (3_500, 6_500)  # the middle
    clips = fill_clips(spans(10_000, 10_000), 6_000)
    assert [x.duration_ms for x in clips] == [4_000, 2_000]


def test_short_tail_is_avoided_by_shortening_the_clip_before() -> None:
    # 5000 ms over 4000 ms shots would leave a 1000 ms tail: fine. 4300 would leave 300.
    clips = fill_clips(spans(4_000, 4_000), 4_300)
    assert [x.duration_ms for x in clips] == [3_500, 800]
    assert sum(x.duration_ms for x in clips) == 4_300


def test_wraps_around_when_shots_run_out() -> None:
    clips = fill_clips(spans(1_000, 1_000), 5_000)
    assert [x.shot_id for x in clips] == ["sh_0", "sh_1", "sh_0", "sh_1", "sh_0"]
    assert sum(x.duration_ms for x in clips) == 5_000


def test_degenerate_inputs() -> None:
    assert fill_clips(spans(1_000), 0) == []
    with pytest.raises(ValueError):
        fill_clips([], 1_000)
    assert [x.duration_ms for x in fill_clips(spans(5_000), 300)] == [300]


durations = st.lists(st.integers(1, 12_000), min_size=1, max_size=12)


@given(durations, st.integers(1, 90_000))
def test_fill_properties(durs: list[int], target: int) -> None:
    cands = spans(*durs)
    by_id = {s.shot_id: s for s in cands}
    clips = fill_clips(cands, target)
    assert sum(c.duration_ms for c in clips) == target
    for c in clips:
        shot = by_id[c.shot_id]
        assert shot.start_ms <= c.src_in_ms < c.src_out_ms <= shot.end_ms
        assert c.duration_ms <= MAX_CLIP_MS


@given(
    st.lists(st.integers(2 * MIN_CLIP_MS, 12_000), min_size=1, max_size=12),
    st.integers(MIN_CLIP_MS, 90_000),
)
def test_no_short_clips_when_shots_are_long_enough(durs: list[int], target: int) -> None:
    clips = fill_clips(spans(*durs), target)
    assert all(c.duration_ms >= MIN_CLIP_MS for c in clips)
