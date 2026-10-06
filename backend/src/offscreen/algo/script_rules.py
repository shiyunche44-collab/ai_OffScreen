"""The rule checker for AI-written scripts (ARCHITECTURE §7.2 ③). Pure: no IO, no models.

One function, `check`, serves three callers: the writing stage (violations go back to the model
as repair instructions), the whole finished script (violations that survive become `rule`
annotations) and, later, saves by a person. Rules:

- total length near the target (`total_length`)
- every segment between `min_segment_chars` and `max_segment_chars` (`segment_length`)
- every segment cites scenes, and only scenes the movie has (`scene_refs`)
- no banned word of the style (`banned_word`)
- people are named as the confirmed characters are (`name`): a text that writes a name one
  character off from a confirmed one ("辛特尔" for "辛忒尔") is a slip, not another person
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from offscreen.algo.script import count_chars
from offscreen.domain.script import Annotation

MIN_SEGMENT_CHARS = 15
MAX_SEGMENT_CHARS = 80
LENGTH_TOLERANCE = 0.15
MIN_NAME_CHARS = 3
"""Shorter names are not checked for slips: one wrong character of two is just another word."""

Rule = Literal["total_length", "segment_length", "scene_refs", "banned_word", "name"]


@dataclass(frozen=True)
class SegmentDraft:
    """A segment as written, before it is known to be valid (so it may lack scene refs)."""

    text: str
    scene_refs: Sequence[str] = ()
    id: str | None = None


@dataclass(frozen=True)
class KnownName:
    """A confirmed character: its name and the other ways the text may call it."""

    name: str
    aliases: Sequence[str] = ()


@dataclass(frozen=True)
class Rules:
    scene_ids: Sequence[str]
    target_chars: int | None = None
    """Total length to aim for; None: do not check the total."""
    tolerance: float = LENGTH_TOLERANCE
    min_segment_chars: int = MIN_SEGMENT_CHARS
    max_segment_chars: int = MAX_SEGMENT_CHARS
    banned_words: Sequence[str] = ()
    names: Sequence[KnownName] = ()
    whole: str = "全文"
    """What the checked segments make up, in the messages ("全文", "本节")."""


@dataclass(frozen=True)
class Violation:
    rule: Rule
    message: str
    """In Chinese, for the model: it names the segment as 第 n 段."""
    segment: int | None = None
    """0-based index of the segment it is about; None: the whole text."""
    segment_id: str | None = None

    def annotation(self, fallback_id: str) -> Annotation:
        return Annotation(
            segment_id=self.segment_id or fallback_id, type="rule", message=self.message
        )


def check(segments: Sequence[SegmentDraft], rules: Rules) -> list[Violation]:
    """Every violation, segment by segment and then the total, in a stable order."""
    valid = set(rules.scene_ids)
    found: list[Violation] = []

    def add(rule: Rule, message: str, index: int | None = None) -> None:
        seg_id = segments[index].id if index is not None else None
        found.append(Violation(rule, message, index, seg_id))

    total = 0
    for i, seg in enumerate(segments):
        n = i + 1
        chars = count_chars(seg.text)
        total += chars
        if not seg.scene_refs:
            add("scene_refs", f"第 {n} 段没有 scene_refs", i)
        unknown = [r for r in seg.scene_refs if r not in valid]
        if unknown:
            add("scene_refs", f"第 {n} 段引用了不存在的场景：{', '.join(unknown)}", i)
        if chars < rules.min_segment_chars:
            add(
                "segment_length",
                f"第 {n} 段只有 {chars} 字，太短（至少 {rules.min_segment_chars} 字）",
                i,
            )
        elif chars > rules.max_segment_chars:
            add(
                "segment_length",
                f"第 {n} 段有 {chars} 字，太长（上限 {rules.max_segment_chars}），请拆成多段",
                i,
            )
        used = [w for w in rules.banned_words if w in seg.text]
        if used:
            add("banned_word", f"第 {n} 段含有禁用词：{', '.join(used)}", i)
        for slip, name in name_slips(seg.text, rules.names):
            add("name", f"第 {n} 段把人物「{name}」写成了「{slip}」", i)

    if rules.target_chars is not None:
        lo = round(rules.target_chars * (1 - rules.tolerance))
        hi = round(rules.target_chars * (1 + rules.tolerance))
        if not lo <= total <= hi:
            advice = "删减" if total > hi else "补充"
            add(
                "total_length",
                f"{rules.whole}共 {total} 字，目标 {rules.target_chars} 字"
                f"（允许 {lo}–{hi}）；请{advice}",
            )
    return found


def name_slips(text: str, names: Sequence[KnownName]) -> list[tuple[str, str]]:
    """`(what the text says, the confirmed name)` for every place where the text has a name of
    a confirmed character with exactly one character replaced.

    Exact names and aliases are never slips, and neither is anything inside them. Names shorter
    than `MIN_NAME_CHARS` are skipped. Latin names are compared ignoring case."""
    allowed = {w for k in names for w in (k.name, *k.aliases) if w}
    covered: list[tuple[int, int]] = []
    for word in allowed:
        start = 0
        while (at := _find(text, word, start)) != -1:
            covered.append((at, at + len(word)))
            start = at + 1

    slips: list[tuple[str, str]] = []
    for known in names:
        name = known.name
        if len(name) < MIN_NAME_CHARS:
            continue
        for at in range(len(text) - len(name) + 1):
            window = text[at : at + len(name)]
            if _same(window, name) or not _one_off(window, name):
                continue
            if any(a < at + len(name) and at < b for a, b in covered):
                continue
            if any(_same(window, w) for w in allowed):
                continue
            slips.append((window, name))
            covered.append((at, at + len(name)))  # one report per place
    return slips


def _same(a: str, b: str) -> bool:
    return a.casefold() == b.casefold()


def _find(text: str, word: str, start: int) -> int:
    return text.casefold().find(word.casefold(), start)


def _one_off(a: str, b: str) -> bool:
    """Equal length, exactly one position differs."""
    return (
        len(a) == len(b)
        and sum(x.casefold() != y.casefold() for x, y in zip(a, b, strict=True)) == 1
    )


def as_annotations(violations: Sequence[Violation], segment_ids: Sequence[str]) -> list[Annotation]:
    """The violations as `rule` annotations of a script. One about the whole text goes on the
    last segment, since an annotation needs a segment."""
    last = segment_ids[-1]
    return [
        v.annotation(segment_ids[v.segment] if v.segment is not None else last) for v in violations
    ]
