"""Duration fitting: cut and speed shots so the clips play for exactly the target. Pure.

Given shots in preference order and a target playback duration, choose how many shots to use
and cut one clip from the middle of each, with a playback speed that makes the durations add
up to `target_ms` to the millisecond. That is within one frame at any frame rate, and the
compiler (`algo.compile`) rounds from cumulative time, so no frame grid is needed here.

Constraints: a clip's source length is 800-4000 ms (a shot shorter than 800 ms is used whole),
and its speed is 0.85-1.15x. A clip therefore plays for `source / speed` ms, anywhere in
`[smin / 1.15, smax / 0.85]` where `smin`/`smax` are the shortest/longest source lengths the
shot allows. The playback range of n clips is the sum of those intervals, so a target is
reachable with n shots exactly when it lies in that sum: the fitter takes the fewest shots
whose sum contains the target, preferring a count that needs no slow-motion (speed >= 1),
and spreads the target over them in proportion to each shot's slack.

Shots are used in the order given and wrap around when there are too few (the same shot is
then cut again, identically). A shot too short to cut is skipped."""

from __future__ import annotations

from dataclasses import dataclass

TimeMs = int

MIN_CLIP_MS = 800
MAX_CLIP_MS = 4000
MIN_SPEED = 0.85
MAX_SPEED = 1.15
MAX_CLIPS = 200
"""Safety bound on the clips of one fit (a 12 s segment needs at most about 18)."""


@dataclass(frozen=True)
class Shot:
    """Candidate shot: where it starts in the source and how much footage it has."""

    shot_id: str
    available_ms: TimeMs
    start_ms: TimeMs = 0


@dataclass(frozen=True)
class Clip:
    """Trimmed clip ready for the plan: absolute source range and playback speed."""

    shot_id: str
    src_in_ms: TimeMs
    src_out_ms: TimeMs
    speed: float  # 1.0 = original, 0.85-1.15 range

    @property
    def source_duration_ms(self) -> TimeMs:
        return self.src_out_ms - self.src_in_ms

    def playback_duration_ms(self) -> TimeMs:
        """Duration at playback speed."""
        return round(self.source_duration_ms / self.speed)


@dataclass(frozen=True)
class _Slot:
    shot: Shot
    smin: int  # shortest / longest source length that may be cut from the shot
    smax: int
    dmin: int  # playback range this shot can cover
    dmax: int

    @property
    def dnat(self) -> int:
        """Longest playback without slow-motion."""
        return self.smax


def fit_duration(shots: list[Shot], target_ms: TimeMs) -> list[Clip]:
    """Clips from `shots` whose playback durations add up to `target_ms` exactly.

    Shots are used in the order given. A few targets fall in a gap that the first shots cannot
    cover (one 0.9 s shot plays for at most 1.06 s, two clips for at least 1.39 s); the fitter
    then tries again with the longest shots first. Raises ValueError when the target is shorter
    than the shortest possible clip, no shot has footage, or even that does not work."""
    if target_ms <= 0:
        raise ValueError("target_ms must be positive")
    usable = [_slot(s) for s in shots if s.available_ms > 0]
    if not usable:
        raise ValueError("no shots available")
    try:
        slots = _choose(usable, target_ms)
    except ValueError as first:
        longest_first = sorted(usable, key=lambda s: -s.shot.available_ms)
        if longest_first == usable:
            raise
        try:
            slots = _choose(longest_first, target_ms)
        except ValueError:
            raise first from None
    lengths = _spread(slots, target_ms)
    return [_cut(slot, d) for slot, d in zip(slots, lengths, strict=True)]


def _slot(shot: Shot) -> _Slot:
    smax = min(shot.available_ms, MAX_CLIP_MS)
    smin = min(MIN_CLIP_MS, smax)
    return _Slot(
        shot=shot,
        smin=smin,
        smax=smax,
        dmin=-(-smin * 100 // round(MAX_SPEED * 100)),  # ceil(smin / 1.15)
        dmax=smax * 100 // round(MIN_SPEED * 100),  # floor(smax / 0.85)
    )


def _choose(usable: list[_Slot], target_ms: TimeMs) -> list[_Slot]:
    """The shots to use, in order: the fewest distinct shots that can reach the target without
    slow-motion, else the fewest clips (repeating shots if need be) that can reach it at all."""
    lo = nat = hi = 0
    reachable: int | None = None
    for n in range(1, MAX_CLIPS + 1):
        slot = usable[(n - 1) % len(usable)]
        lo += slot.dmin
        nat += slot.dnat
        hi += slot.dmax
        if lo > target_ms:
            break
        if n <= len(usable) and target_ms <= nat:
            return [usable[i % len(usable)] for i in range(n)]
        if reachable is None and target_ms <= hi:
            reachable = n
    if reachable is None:
        raise ValueError(
            f"cannot fill {target_ms}ms: the shortest clip plays for {usable[0].dmin}ms"
            if target_ms < usable[0].dmin
            else f"cannot fill {target_ms}ms with the given shots"
        )
    return [usable[i % len(usable)] for i in range(reachable)]


def _spread(slots: list[_Slot], target_ms: TimeMs) -> list[int]:
    """Playback durations, each within its slot's range, adding up to `target_ms`. The slack
    above every minimum is shared in proportion to how much each slot can take (largest
    remainder, so the integers add up exactly)."""
    upper = [s.dnat if target_ms <= sum(t.dnat for t in slots) else s.dmax for s in slots]
    lower = [s.dmin for s in slots]
    room = [u - lo for u, lo in zip(upper, lower, strict=True)]
    extra = target_ms - sum(lower)
    total_room = sum(room)
    shares = [extra * r // total_room if total_room else 0 for r in room]
    left = extra - sum(shares)
    by_remainder = sorted(
        range(len(slots)),
        key=lambda i: (-((extra * room[i]) % total_room if total_room else 0), i),
    )
    for i in by_remainder:
        if left == 0:
            break
        if shares[i] < room[i]:
            shares[i] += 1
            left -= 1
    return [lo + sh for lo, sh in zip(lower, shares, strict=True)]


def _cut(slot: _Slot, playback_ms: int) -> Clip:
    """The middle of the shot, as long as `playback_ms` allows at a speed within limits."""
    source_ms = min(max(playback_ms, slot.smin), slot.smax)
    offset = (slot.shot.available_ms - source_ms) // 2
    start = slot.shot.start_ms + offset
    return Clip(
        shot_id=slot.shot.shot_id,
        src_in_ms=start,
        src_out_ms=start + source_ms,
        speed=source_ms / playback_ms,
    )
