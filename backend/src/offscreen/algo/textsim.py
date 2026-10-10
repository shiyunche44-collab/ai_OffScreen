"""Text similarity for Chinese and mixed text, which has no spaces to split words on. Pure."""

from __future__ import annotations

import re

_NOISE = re.compile(r"[\s\W_]+", re.UNICODE)


def bigrams(text: str) -> set[str]:
    """Character bigrams of `text` without spaces and punctuation, lower-cased. A text of one
    character yields that character."""
    chars = _NOISE.sub("", text).lower()
    if len(chars) < 2:
        return {chars} if chars else set()
    return {chars[i : i + 2] for i in range(len(chars) - 1)}


def bigram_similarity(a: str, b: str) -> float:
    """Dice coefficient of the bigram sets, in [0, 1]; 0 when either text has no letters."""
    x, y = bigrams(a), bigrams(b)
    if not x or not y:
        return 0.0
    return 2 * len(x & y) / (len(x) + len(y))


def coverage(description: str, text: str) -> float:
    """Share of the description's bigrams that occur in `text`, in [0, 1].

    A shot description is short and the narration long, so Dice would always be small;
    coverage asks how much of what the shot shows the text talks about."""
    x, y = bigrams(description), bigrams(text)
    if not x:
        return 0.0
    return len(x & y) / len(x)
