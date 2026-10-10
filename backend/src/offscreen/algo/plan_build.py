"""Incremental plan building: what to redo when the script changed, and what a lock keeps. Pure.

A narration segment of the previous plan is reused as it is when the script text it was made from
(`text_hash`), the voice and its audio file are all unchanged; otherwise it is rebuilt: new
voice-over, new footage. Footage a person locked (`Clip.locked`) is kept in a rebuild; new
footage fills only what the locked clips leave of the new voice-over, and takes the place of the
first unlocked clip. Segments are matched by id, so adding, removing or moving a segment in the
script leaves the others reusable."""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Literal

from offscreen.domain.plan import Clip, EditPlan, PlanSegment, VoiceSpec

Reason = Literal["new", "text", "voice", "stale", "audio_missing", "unhashed"]


def text_digest(text: str) -> str:
    """The `text_hash` of a narration text (`PlanSegment.text_hash`)."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Decision:
    """`reason` is None for a reused segment."""

    segment_id: str
    reason: Reason | None

    @property
    def reuse(self) -> bool:
        return self.reason is None


def decide_narration(
    segment_id: str,
    text: str,
    voice: VoiceSpec,
    previous: PlanSegment | None,
    *,
    audio_exists: bool,
) -> Decision:
    """Whether the segment of the previous plan can stay, and if not, why. The first reason
    that applies is given. `voice` is the one the segment should have (its own when a person
    pinned it, else the script's)."""
    if previous is None or previous.kind != "narration":
        return Decision(segment_id, "new")
    if previous.stale:
        return Decision(segment_id, "stale")
    if previous.text_hash is None:
        return Decision(segment_id, "unhashed")
    if previous.text_hash != text_digest(text):
        return Decision(segment_id, "text")
    if previous.voice != voice:
        return Decision(segment_id, "voice")
    if previous.audio is None or not audio_exists:
        return Decision(segment_id, "audio_missing")
    return Decision(segment_id, None)


def voice_for(previous: PlanSegment | None, default: VoiceSpec) -> VoiceSpec:
    """A person's choice of voice for a segment outlives rebuilds; otherwise the script's."""
    if previous is not None and previous.kind == "narration" and previous.voice_pinned:
        return previous.voice or default
    return default


def plan_order(
    script_ids: Sequence[str],
    previous: EditPlan | None,
    previous_script_ids: Collection[str] | None,
) -> list[str]:
    """The segment ids of the new plan, in order.

    Without a previous plan, or without the script it was built from, the plan is the script.
    Otherwise a person's structural edits of the plan stand: the plan's own order is kept,
    a segment the person deleted (it was in the previous script, and is not in the plan) stays
    deleted, and an original-sound segment the person inserted (it is in neither script) stays.
    Segments the script gained since are put after the script segment before them."""
    if previous is None or previous_script_ids is None:
        return list(script_ids)
    current, known = set(script_ids), set(previous_script_ids)
    order = [
        s.id
        for s in previous.segments
        if s.id in current or (s.id not in known and s.kind == "original")
    ]
    present = set(order)
    for i, sid in enumerate(script_ids):
        if sid in present or sid in known:
            continue
        at = next(
            (order.index(p) + 1 for p in reversed(script_ids[:i]) if p in present),
            0,
        )
        order.insert(at, sid)
        present.add(sid)
    return order


def playback_ms(clip: Clip) -> int:
    return round((clip.src_out_ms - clip.src_in_ms) / clip.speed)


@dataclass(frozen=True)
class LockedClips:
    """The locked clips of a previous segment and what is left to fill around them."""

    clips: list[Clip]
    insert_at: int
    """Index in `clips` where new footage goes: where the first unlocked clip was."""
    remaining_ms: int
    """Voice-over still without footage; 0 or less when the locked clips cover it."""


def keep_locked(previous: Sequence[Clip], audio_ms: int) -> LockedClips:
    locked = [c for c in previous if c.locked]
    first_free = next((i for i, c in enumerate(previous) if not c.locked), None)
    insert_at = (
        len(locked) if first_free is None else sum(1 for c in previous[:first_free] if c.locked)
    )
    return LockedClips(locked, insert_at, audio_ms - sum(playback_ms(c) for c in locked))


def with_fresh_footage(kept: LockedClips, fresh: Sequence[Clip]) -> list[Clip]:
    return [*kept.clips[: kept.insert_at], *fresh, *kept.clips[kept.insert_at :]]


def same_content(a: EditPlan, b: EditPlan) -> bool:
    """Whether two plans say the same, whatever their place in the history (id, version, parent,
    author)."""
    skip = {"id", "version", "parent_version", "author"}
    return a.model_dump(mode="json", exclude=skip) == b.model_dump(mode="json", exclude=skip)


@dataclass
class PlanReport:
    """What a build did, for the job log and the artifact's manifest."""

    rebuilt: dict[str, Reason] = field(default_factory=dict)
    reused: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [f"plan: {len(self.reused)} reused, {len(self.rebuilt)} rebuilt"]
        out += [f"plan: rebuilt {sid} ({why})" for sid, why in self.rebuilt.items()]
        out += [f"plan: reused {sid}" for sid in self.reused]
        out += [f"plan: removed {sid} (no longer in the script)" for sid in self.removed]
        out += [f"plan: warning: {w}" for w in self.warnings]
        return out
