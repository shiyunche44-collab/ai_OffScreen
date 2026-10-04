"""Length control and rule checks for AI-written scripts. Pure: no IO, no models."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from offscreen.domain.script import OutlineBeat, ScriptSegment

DEFAULT_CHARS_PER_S = 4.5
"""Spoken characters per second for narration; calibrated per voice in M6."""
LENGTH_TOLERANCE = 0.15
MIN_SEGMENT_CHARS = 5
MAX_SEGMENT_CHARS = 200


def count_chars(text: str) -> int:
    """Spoken characters: letters and digits, so spaces and punctuation do not count."""
    return sum(1 for c in text if unicodedata.category(c)[0] in "LN")


def target_chars(target_s: int, chars_per_s: float) -> int:
    return round(target_s * chars_per_s)


def estimate_duration_s(chars: int, chars_per_s: float) -> float:
    return chars / chars_per_s


def check_draft(
    texts: Sequence[str],
    scene_refs: Sequence[Sequence[str]],
    valid_scene_ids: Sequence[str],
    *,
    target: int,
    tolerance: float = LENGTH_TOLERANCE,
) -> list[str]:
    """Rule violations of a draft (one entry per segment), as messages to show the model.

    Checks: scene refs exist, each segment's length is sane, and the total length is within
    `tolerance` of `target` characters."""
    valid = set(valid_scene_ids)
    errors: list[str] = []
    total = 0
    for n, (text, refs) in enumerate(zip(texts, scene_refs, strict=True), 1):
        n_chars = count_chars(text)
        total += n_chars
        unknown = [r for r in refs if r not in valid]
        if unknown:
            errors.append(f"第 {n} 段引用了不存在的场景：{', '.join(unknown)}")
        if not refs:
            errors.append(f"第 {n} 段没有 scene_refs")
        if n_chars < MIN_SEGMENT_CHARS:
            errors.append(f"第 {n} 段只有 {n_chars} 字，太短")
        elif n_chars > MAX_SEGMENT_CHARS:
            errors.append(f"第 {n} 段有 {n_chars} 字，太长（上限 {MAX_SEGMENT_CHARS}），请拆成多段")
    lo, hi = round(target * (1 - tolerance)), round(target * (1 + tolerance))
    if not lo <= total <= hi:
        advice = "删减" if total > hi else "补充"
        errors.append(f"全文共 {total} 字，目标 {target} 字（允许 {lo}–{hi}）；请{advice}")
    return errors


def build_outline(segments: Sequence[ScriptSegment], chars_per_s: float) -> list[OutlineBeat]:
    """One outline entry per run of consecutive segments sharing a beat, with the scenes
    they cite (in first-use order) and the narration time they take."""
    runs: list[tuple[str, list[str], int]] = []
    for seg in segments:
        if seg.kind != "narration":
            continue
        beat = seg.beat or "body"
        if runs and runs[-1][0] == beat:
            _, refs, chars = runs[-1]
            refs.extend(r for r in seg.scene_refs if r not in refs)
            runs[-1] = (beat, refs, chars + count_chars(seg.text))
        else:
            runs.append((beat, list(dict.fromkeys(seg.scene_refs)), count_chars(seg.text)))
    return [
        OutlineBeat(beat=b, scene_refs=refs, target_s=max(1, round(chars / chars_per_s)))
        for b, refs, chars in runs
    ]
