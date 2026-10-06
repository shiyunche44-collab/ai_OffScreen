"""Measuring shot detection against hand-marked cuts. Pure: frames in, numbers out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from offscreen.domain.common import Rational

DEFAULT_TOLERANCE = 2
"""A detected cut counts as right when it is within this many frames of a marked one."""


@dataclass(frozen=True)
class CutScore:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: tuple[int, ...]
    """Detected cuts that match no marked cut (frames)."""
    false_negatives: tuple[int, ...]
    """Marked cuts nothing detected (frames)."""
    matches: tuple[tuple[int, int], ...]
    """`(detected, marked)` pairs."""


def shot_cut_frames(starts_ms: Sequence[int], fps: Rational) -> list[int]:
    """The cut frames of a list of shot start times: every shot after the first begins at one."""
    frames = sorted({fps.frames_for_ms(ms) for ms in starts_ms})
    return [f for f in frames if f >= 1]


def evaluate_cuts(
    detected: Sequence[int], marked: Sequence[int], tolerance: int = DEFAULT_TOLERANCE
) -> CutScore:
    """Precision, recall and F1 of `detected` against `marked`, both ascending frame numbers.
    Each marked cut is claimed by at most one detected cut (the nearest within `tolerance`, pairs
    taken in order of closeness), so a cut found twice counts once and its twin is a false
    positive. No detections and no marks is a perfect score."""
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    pairs = sorted(
        (abs(d - m), i, j)
        for i, d in enumerate(detected)
        for j, m in enumerate(marked)
        if abs(d - m) <= tolerance
    )
    used_d: set[int] = set()
    used_m: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used_d or j in used_m:
            continue
        used_d.add(i)
        used_m.add(j)
        matched.append((detected[i], marked[j]))
    tp = len(matched)
    fp = tuple(d for i, d in enumerate(detected) if i not in used_d)
    fn = tuple(m for j, m in enumerate(marked) if j not in used_m)
    precision = tp / len(detected) if detected else (1.0 if not marked else 0.0)
    recall = tp / len(marked) if marked else (1.0 if not detected else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return CutScore(precision, recall, f1, tp, fp, fn, tuple(sorted(matched)))
