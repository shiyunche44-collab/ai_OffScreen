from __future__ import annotations

from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.story import check_story_refs, chunk_lines, fmt_clock, plan_scenes
from offscreen.domain.index import TranscriptLine


def mk(spans: list[tuple[int, int, str]]) -> list[TranscriptLine]:
    return [
        TranscriptLine(id=f"ln_{i:04d}", start_ms=a, end_ms=b, text=t)
        for i, (a, b, t) in enumerate(spans, 1)
    ]


def test_fmt_clock() -> None:
    assert fmt_clock(0) == "00:00"
    assert fmt_clock(61_999) == "01:01"
    assert fmt_clock(3_725_000) == "1:02:05"


def test_chunk_by_chars_and_pause() -> None:
    lines = mk([(0, 1000, "aaaa"), (1500, 2500, "bbbb"), (9000, 10000, "cc"), (10500, 11000, "d")])
    # 4 + 4 > 7 -> break before "bbbb"; then "bbbb" is over half full and a 6.5 s pause
    # precedes "cc" -> break again; "cc" + "d" fit.
    assert chunk_lines(lines, max_chars=7, max_span_ms=10**9) == [(0, 1), (1, 2), (2, 4)]
    # Half full (>= 4 of 8) and a >= 4 s pause before line 3 -> break there.
    assert chunk_lines(lines, max_chars=8, max_span_ms=10**9) == [(0, 2), (2, 4)]


def test_chunk_by_span() -> None:
    lines = mk([(0, 1000, "a"), (3000, 4000, "b"), (5000, 6000, "c")])
    assert chunk_lines(lines, max_chars=10**6, max_span_ms=5000) == [(0, 2), (2, 3)]


def test_chunk_empty() -> None:
    assert chunk_lines([], max_chars=10, max_span_ms=10) == []


def test_scenes_snap_to_shot_starts() -> None:
    lines = mk([(0, 1000, "aaaa"), (1500, 2500, "bbbb"), (9000, 10000, "cc"), (10500, 11000, "d")])
    shot_starts = [0, 3000, 6000, 8000, 11000]
    plans = plan_scenes(lines, shot_starts, max_chars=8, max_span_ms=10**9)
    # The pause 2500..9000 has midpoint 5750 -> nearest shot start 6000 (index 2).
    assert [(p.line_lo, p.line_hi, p.shot_lo, p.shot_hi) for p in plans] == [
        (0, 2, 0, 2),
        (2, 4, 2, 5),
    ]


def test_scenes_merge_when_border_collides() -> None:
    # One shot only: every border snaps to 0 or past the end, so everything is one scene.
    lines = mk([(0, 1000, "aaaa"), (9000, 10000, "bbbb")])
    plans = plan_scenes(lines, [0], max_chars=4, max_span_ms=10**9)
    assert [(p.line_lo, p.line_hi, p.shot_lo, p.shot_hi) for p in plans] == [(0, 2, 0, 1)]


def test_scenes_empty_inputs() -> None:
    assert plan_scenes([], [0, 10], max_chars=10, max_span_ms=10) == []
    assert plan_scenes(mk([(0, 1, "a")]), [], max_chars=10, max_span_ms=10) == []


@st.composite
def film(draw: st.DrawFn) -> tuple[list[TranscriptLine], list[int]]:
    n = draw(st.integers(1, 40))
    gaps = draw(st.lists(st.integers(0, 20_000), min_size=n, max_size=n))
    durs = draw(st.lists(st.integers(1, 5_000), min_size=n, max_size=n))
    texts = draw(st.lists(st.text(min_size=1, max_size=30), min_size=n, max_size=n))
    spans, t = [], 0
    for g, d, tx in zip(gaps, durs, texts, strict=True):
        spans.append((t + g, t + g + d, tx))
        t += g + d
    n_shots = draw(st.integers(1, 30))
    starts = sorted(draw(st.sets(st.integers(1, max(2, t + 1_000)), max_size=n_shots - 1)) | {0})
    return mk(spans), starts


@given(film(), st.integers(1, 200), st.integers(1_000, 120_000))
def test_scenes_partition_lines_and_shots(
    f: tuple[list[TranscriptLine], list[int]], max_chars: int, max_span: int
) -> None:
    lines, starts = f
    plans = plan_scenes(lines, starts, max_chars=max_chars, max_span_ms=max_span)
    assert plans
    assert plans[0].line_lo == 0 and plans[-1].line_hi == len(lines)
    assert plans[0].shot_lo == 0 and plans[-1].shot_hi == len(starts)
    for p in plans:
        assert p.line_lo < p.line_hi and p.shot_lo < p.shot_hi
    for a, b in pairwise(plans):
        assert a.line_hi == b.line_lo and a.shot_hi == b.shot_lo


def test_check_story_refs() -> None:
    valid = ["sc_001", "sc_002", "sc_003"]
    assert check_story_refs([["sc_001"], ["sc_002", "sc_003"]], ["sc_002"], valid) == []
    errs = check_story_refs([["sc_001", "sc_009"], ["sc_001"]], ["sc_010"], valid)
    assert len(errs) == 3
    assert "sc_009" in errs[0] and "sc_001" in errs[1] and "sc_010" in errs[2]
