"""TTS helpers: the cache key and alignment of engine timestamps to input characters. Pure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WordSpan:
    """Engine timestamps for `text[char_lo:char_hi]`, in (fractional) milliseconds."""

    start_ms: float
    end_ms: float
    char_lo: int
    char_hi: int


def tts_cache_key(text: str, voice_id: str, speed: float, engine_id: str) -> str:
    """`sha256:<hex>` over everything that changes the audio (ARCHITECTURE §7.3)."""
    payload = json.dumps(
        {"text": text, "voice": voice_id, "speed": speed, "engine": engine_id},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def char_timings_from_spans(
    n_chars: int, spans: Sequence[WordSpan], duration_ms: int
) -> list[tuple[int, int]]:
    """One `(start_ms, end_ms)` per character of the input, from whatever granularity the
    engine reported.

    Engines time words, not characters: one entry per CJK character but sub-word pieces for
    English, and a number may be repeated once per spoken syllable with the same offsets.
    Entries with identical offsets are merged into one span, and a span covering several
    characters is divided evenly among them. Characters no span covers (spaces, tags) get an
    empty interval where the previous one ended. The result is clamped to the audio and is
    monotonic: starts never precede the previous end."""
    if n_chars <= 0:
        return []
    merged: dict[tuple[int, int], tuple[float, float]] = {}
    for sp in spans:
        lo, hi = max(0, sp.char_lo), min(n_chars, sp.char_hi)
        if lo >= hi:
            continue
        a, b = merged.get((lo, hi), (sp.start_ms, sp.end_ms))
        merged[(lo, hi)] = (min(a, sp.start_ms), max(b, sp.end_ms))

    times: list[tuple[float, float] | None] = [None] * n_chars
    for (lo, hi), (a, b) in sorted(merged.items()):
        for k in range(lo, hi):
            if times[k] is None:  # the first (earliest, then shortest) span claiming a char wins
                times[k] = (
                    a + (b - a) * (k - lo) / (hi - lo),
                    a + (b - a) * (k - lo + 1) / (hi - lo),
                )

    out: list[tuple[int, int]] = []
    prev_end = 0
    for t in times:
        start, end = (float(prev_end), float(prev_end)) if t is None else t
        s = min(max(round(start), prev_end), duration_ms)
        e = min(max(round(end), s), duration_ms)
        out.append((s, e))
        prev_end = e
    return out


WORD_SLACK = 8
"""Characters a reported word may start after the previous one (spaces, punctuation, a
skipped word)."""


def spans_from_words(text: str, words: Sequence[tuple[str, float, float]]) -> list[WordSpan]:
    """Place `(word, start_ms, end_ms)` entries, reported as text without offsets, on the
    characters of `text`: each is looked for after the previous one. Engines attach
    punctuation to the word before it and may drop spaces, so a word is matched without
    surrounding spaces, and failing that without its punctuation. A word is only looked for
    within `WORD_SLACK` characters of where the previous one ended, so one the engine rewrote
    cannot latch onto a later occurrence; it is skipped and its characters are left to
    `char_timings_from_spans`."""
    spans: list[WordSpan] = []
    cursor = 0
    for word, start, end in words:
        for probe in (word.strip(), _core(word)):
            if not probe:
                continue
            at = text.find(probe, cursor, cursor + WORD_SLACK + len(probe))
            if at >= 0:
                spans.append(WordSpan(start, end, at, at + len(probe)))
                cursor = at + len(probe)
                break
    return spans


def _core(word: str) -> str:
    """The word without the punctuation and spaces around it."""
    keep = [i for i, c in enumerate(word) if c.isalnum()]
    return word[keep[0] : keep[-1] + 1] if keep else ""
