"""Planning the shot-description requests (ARCHITECTURE §7.1, captions). Pure: no IO, no models.

Shots go to the vision model in consecutive batches; each shot comes with the dialogue spoken
around it. Token use is estimated up front, so a long film's cost is known before the first call."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from typing import TypeVar

from offscreen.domain.index import TranscriptLine

T = TypeVar("T")

DIALOGUE_PAD_MS = 2_000
"""Dialogue this close to a shot (before its start, after its end) is shown with it."""
MAX_DIALOGUE_CHARS = 160
"""Per shot; the rest is cut, the model only needs the gist of what is being said."""

IMAGE_TOKENS = 500
"""Rough input cost of one 640x360 frame. Deliberately on the high side: an estimate that is too
low would let a long film surprise the subscription's quota window."""
TOKENS_PER_CJK_CHAR = 1.0
REQUEST_OVERHEAD_TOKENS = 700
"""Instructions + schema text of one request."""
OUT_TOKENS_PER_SHOT = 90


def batches(items: Sequence[T], size: int) -> list[list[T]]:
    """Consecutive batches of at most `size` items, covering every item once, in order."""
    if size < 1:
        raise ValueError("batch size must be at least 1")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def dialogue_near(
    lines: Sequence[TranscriptLine],
    start_ms: int,
    end_ms: int,
    *,
    pad_ms: int = DIALOGUE_PAD_MS,
    max_chars: int = MAX_DIALOGUE_CHARS,
) -> str:
    """The lines overlapping `[start_ms - pad, end_ms + pad)`, joined with spaces and cut to
    `max_chars` (with an ellipsis). Empty when nobody speaks there."""
    lo, hi = start_ms - pad_ms, end_ms + pad_ms
    text = " ".join(
        x.text.strip() for x in lines if x.start_ms < hi and x.end_ms > lo and x.text.strip()
    )
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def estimate_tokens(
    shot_count: int, *, batch_size: int, frames_per_shot: int, dialogue_chars: int
) -> int:
    """Input + output tokens for describing `shot_count` shots, `dialogue_chars` of dialogue in
    total, in requests of `batch_size` shots."""
    if shot_count <= 0:
        return 0
    requests = ceil(shot_count / batch_size)
    inputs = (
        shot_count * frames_per_shot * IMAGE_TOKENS
        + ceil(dialogue_chars * TOKENS_PER_CJK_CHAR)
        + requests * REQUEST_OVERHEAD_TOKENS
    )
    return inputs + shot_count * OUT_TOKENS_PER_SHOT


def missing(requested: Sequence[str], got: Sequence[str]) -> list[str]:
    """Requested ids the reply did not cover, in request order."""
    have = set(got)
    return [r for r in requested if r not in have]
