"""Character timings for synthesized speech the engine gave no timestamps for, from what a
speech recognizer heard in it (ARCHITECTURE §7.3). Pure: transcription happens in the plan stage.

The recognizer's words are spread evenly over their characters and matched to the spoken
characters of the text in order (a longest-matching-blocks diff, so a misheard word only loses
its own characters). A matched character takes the recognized time; a run that was not matched
(misheard, written another way) is spread over the time between its matched neighbours. Digits
the recognizer writes as numerals are read out first (`algo.textnorm`), as the text sent to the
engine was."""

from __future__ import annotations

import statistics
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from offscreen.algo.textnorm import normalize

MIN_COVERAGE = 0.6
"""Below this share of matched spoken characters the result is not trusted (wrong language,
the recognizer heard little): no timings is better than wrong ones."""

DEFAULT_CHAR_MS = 220
"""Time per character when no matched character says otherwise (about 4.5 characters/s)."""


@dataclass(frozen=True)
class Alignment:
    char_timings: list[tuple[int, int]]
    """`(start_ms, end_ms)` per character of the text; empty when nothing matched."""
    coverage: float
    """Share of the text's spoken characters matched to recognized ones (0..1)."""

    @property
    def trusted(self) -> bool:
        return bool(self.char_timings) and self.coverage >= MIN_COVERAGE


def spoken(c: str) -> bool:
    """Letters and digits are spoken; spaces and punctuation are not (as `count_chars`)."""
    return unicodedata.category(c)[0] in "LN"


def _key(c: str) -> str:
    return unicodedata.normalize("NFKC", c).lower()


def recognized_chars(words: Sequence[tuple[str, int, int]]) -> list[tuple[str, float, float]]:
    """Each spoken character the recognizer wrote, with an even share of its word's time.
    Numerals are read out first, as the synthesized text had them."""
    out: list[tuple[str, float, float]] = []
    for word, start, end in words:
        chars = [c for c in normalize(word).text if spoken(c)]
        if not chars:
            continue
        step = (end - start) / len(chars)
        out += [(_key(c), start + k * step, start + (k + 1) * step) for k, c in enumerate(chars)]
    return out


def align_chars(text: str, words: Sequence[tuple[str, int, int]], duration_ms: int) -> Alignment:
    """Timings for every character of `text` from recognized `(word, start_ms, end_ms)`."""
    positions = [i for i, c in enumerate(text) if spoken(c)]
    heard = recognized_chars(words)
    if not positions or not heard or duration_ms <= 0:
        return Alignment([], 0.0)

    matcher = SequenceMatcher(
        None, [_key(text[i]) for i in positions], [c for c, _, _ in heard], autojunk=False
    )
    times: list[tuple[float, float] | None] = [None] * len(positions)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            _, s, e = heard[block.b + k]
            times[block.a + k] = (s, e)
    matched = [t for t in times if t is not None]
    if not matched:
        return Alignment([], 0.0)
    char_ms = statistics.median(e - s for s, e in matched) or DEFAULT_CHAR_MS
    _fill_gaps(times, char_ms, duration_ms)

    out: list[tuple[int, int]] = []
    prev = 0
    timed = iter(times)
    for c in text:
        if not spoken(c):
            out.append((prev, prev))
            continue
        t = next(timed)
        assert t is not None
        s = min(max(round(t[0]), prev), duration_ms)
        e = min(max(round(t[1]), s), duration_ms)
        out.append((s, e))
        prev = e
    return Alignment(out, len(matched) / len(positions))


def _fill_gaps(times: list[tuple[float, float] | None], char_ms: float, duration_ms: int) -> None:
    """Spread each run of unmatched characters over the time its matched neighbours leave;
    a run at either end gets `char_ms` per character, within the audio."""
    k = 0
    n = len(times)
    while k < n:
        if times[k] is not None:
            k += 1
            continue
        j = k
        while j < n and times[j] is None:
            j += 1
        before = times[k - 1] if k > 0 else None
        after = times[j] if j < n else None
        count = j - k
        if before is not None and after is not None:
            lo, hi = before[1], max(after[0], before[1])
        elif after is not None:
            hi = after[0]
            lo = max(0.0, hi - count * char_ms)
        else:
            assert before is not None
            lo = before[1]
            hi = min(float(duration_ms), lo + count * char_ms)
        step = (hi - lo) / count
        for m in range(count):
            times[k + m] = (lo + m * step, lo + (m + 1) * step)
        k = j
