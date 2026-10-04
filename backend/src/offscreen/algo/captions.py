"""Subtitle lines from narration text and its per-character timings. Pure.

Lines break after sentence-ending punctuation, then at commas when a line would exceed the
limit, then (for a long unpunctuated run) into near-equal pieces. The limit counts spoken
characters: punctuation and spaces do not."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

HARD_BREAKS = frozenset("。！？!?；;…\n")
SOFT_BREAKS = frozenset("，、,：:—")


@dataclass(frozen=True)
class CaptionLine:
    text: str
    start_ms: int
    end_ms: int


def _spoken(c: str) -> bool:
    return unicodedata.category(c)[0] in "LN"


def _clauses(text: str) -> list[tuple[int, int, bool]]:
    """`(lo, hi, ends_sentence)` ranges of `text`, each ending after a break character."""
    out: list[tuple[int, int, bool]] = []
    lo = 0
    for i, c in enumerate(text):
        if c in HARD_BREAKS or c in SOFT_BREAKS:
            out.append((lo, i + 1, c in HARD_BREAKS))
            lo = i + 1
    if lo < len(text):
        out.append((lo, len(text), True))
    return out


def _split_long(text: str, lo: int, hi: int, limit: int) -> list[tuple[int, int]]:
    """Cut a run with more than `limit` spoken characters into near-equal pieces."""
    idx = [i for i in range(lo, hi) if _spoken(text[i])]
    pieces = -(-len(idx) // limit)
    size = -(-len(idx) // pieces)
    cuts = [idx[k * size] for k in range(1, pieces)]
    bounds = [lo, *cuts, hi]
    return list(pairwise(bounds))


def split_caption_lines(
    text: str,
    char_timings: Sequence[tuple[int, int]],
    duration_ms: int,
    max_chars: int,
) -> list[CaptionLine]:
    """Timed subtitle lines for `text`.

    `char_timings` has one `(start_ms, end_ms)` per character of `text`; when it is empty
    (the engine gave none) the spoken characters share `duration_ms` evenly."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if char_timings and len(char_timings) != len(text):
        raise ValueError("char_timings must have one entry per character of text")
    spoken = [i for i, c in enumerate(text) if _spoken(c)]
    if not spoken:
        return []
    timings = list(char_timings)
    if not timings:
        step = duration_ms / len(spoken)
        timings = [(0, 0)] * len(text)
        for k, i in enumerate(spoken):
            timings[i] = (round(k * step), round((k + 1) * step))

    pieces: list[tuple[int, int, bool]] = []  # (lo, hi, ends_sentence)
    for lo, hi, ends in _clauses(text):
        n = sum(1 for i in range(lo, hi) if _spoken(text[i]))
        if n > max_chars:
            parts = _split_long(text, lo, hi, max_chars)
            pieces += [(a, b, ends and b == hi) for a, b in parts]
        elif n:
            pieces.append((lo, hi, ends))

    ranges: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    cur_n = 0
    for lo, hi, ends in pieces:
        n = sum(1 for i in range(lo, hi) if _spoken(text[i]))
        if cur is not None and cur_n + n > max_chars:
            ranges.append(cur)
            cur, cur_n = None, 0
        cur = (cur[0], hi) if cur else (lo, hi)
        cur_n += n
        if ends:
            ranges.append(cur)
            cur, cur_n = None, 0
    if cur is not None:
        ranges.append(cur)

    lines: list[CaptionLine] = []
    for lo, hi in ranges:
        idx = [i for i in range(lo, hi) if _spoken(text[i])]
        if not idx:
            continue
        first, last = idx[0], idx[-1]
        lines.append(
            CaptionLine(
                text=text[first : last + 1].strip(),
                start_ms=timings[first][0],
                end_ms=max(timings[last][1], timings[first][0]),
            )
        )
    return lines
