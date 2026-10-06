"""Length control and rule checks for AI-written scripts. Pure: no IO, no models."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from offscreen.domain.index import Scene, TranscriptLine
from offscreen.domain.script import OutlineBeat, ScriptSegment

DEFAULT_CHARS_PER_S = 4.5
"""Spoken characters per second for narration; calibrated per voice in M6."""
BEAT_TOLERANCE = 0.2
"""How far one beat's text may stray from its share of the target length."""
KEY_LINES_PER_SCENE = 3
KEY_LINE_CHARS = 40


def count_chars(text: str) -> int:
    """Spoken characters: letters and digits, so spaces and punctuation do not count."""
    return sum(1 for c in text if unicodedata.category(c)[0] in "LN")


def target_chars(target_s: int, chars_per_s: float) -> int:
    return round(target_s * chars_per_s)


def estimate_duration_s(chars: int, chars_per_s: float) -> float:
    return chars / chars_per_s


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


def scene_lines(scene: Scene, lines: Sequence[TranscriptLine]) -> list[TranscriptLine]:
    """The dialogue of a scene: the lines it lists, else the lines that overlap its time range."""
    if scene.line_ids:
        wanted = set(scene.line_ids)
        return [ln for ln in lines if ln.id in wanted]
    return [ln for ln in lines if ln.start_ms < scene.end_ms and ln.end_ms > scene.start_ms]


def key_lines(
    lines: Sequence[TranscriptLine],
    limit: int = KEY_LINES_PER_SCENE,
    max_chars: int = KEY_LINE_CHARS,
) -> list[str]:
    """The most telling lines of a scene, in time order: the longest ones (short interjections
    say little), each cut to `max_chars`."""
    ranked = sorted(range(len(lines)), key=lambda i: (-count_chars(lines[i].text), i))[:limit]
    out: list[str] = []
    for i in sorted(ranked):
        text = " ".join(lines[i].text.split())
        if text:
            out.append(text if len(text) <= max_chars else text[: max_chars - 1] + "…")
    return out
