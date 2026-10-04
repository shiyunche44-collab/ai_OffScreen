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
