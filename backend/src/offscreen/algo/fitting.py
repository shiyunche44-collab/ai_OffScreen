"""Duration fitting: cut and speed clips to match target duration exactly.

Given a list of candidate shots and a target duration (in milliseconds),
select and trim clips from the candidates so their total duration matches
the target exactly, frame-perfect, accounting for playback speed.

Constraints:
- Individual clips: 800–4000 ms at 1x speed
- Speed range: 0.85–1.15x
- Total duration: precise to frame (no floating-point error)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

TimeMs = int
FrameCount = int


class Rational(NamedTuple):
    """Frame rate as num/den (e.g., 30/1 for 30 fps)."""

    num: int
    den: int

    @property
    def inverse(self) -> Rational:
        """Returns den/num (used for ms -> frame conversion)."""
        return Rational(self.den, self.num)


class Duration(NamedTuple):
    """Duration with both millisecond and frame representations."""

    ms: TimeMs  # source time
    frames: FrameCount  # at given frame rate


@dataclass(frozen=True)
class Shot:
    """Candidate shot: its id and available duration."""

    shot_id: str
    available_ms: TimeMs


@dataclass(frozen=True)
class Clip:
    """Trimmed clip ready for timeline: source range and playback speed."""

    shot_id: str
    src_in_ms: TimeMs
    src_out_ms: TimeMs
    speed: float  # 1.0 = original, 0.85–1.15 range

    @property
    def source_duration_ms(self) -> TimeMs:
        return self.src_out_ms - self.src_in_ms

    def playback_duration_ms(self) -> TimeMs:
        """Duration at playback speed (before frame rounding)."""
        return round(self.source_duration_ms / self.speed)


MIN_CLIP_MS = 800
MAX_CLIP_MS = 4000
MIN_SPEED = 0.85
MAX_SPEED = 1.15


def fit_duration(shots: list[Shot], target_ms: TimeMs, fps: Rational | None = None) -> list[Clip]:
    """Fit shots to exact target duration by trimming and speed adjustment.

    Args:
        shots: candidate shots in preference order (best first)
        target_ms: exact target duration in milliseconds
        fps: frame rate (default 30/1); used to verify frame-exact duration

    Returns:
        List of Clip objects whose playback duration equals target_ms,
        or raises ValueError if impossible to fit.

    Raises:
        ValueError: if target cannot be achieved with given shots
    """
    if fps is None:
        fps = Rational(30, 1)

    if target_ms <= 0:
        raise ValueError("target_ms must be positive")

    if not shots:
        raise ValueError("no shots available")

    clips: list[Clip] = []
    accumulated_ms = 0
    shot_idx = 0

    while accumulated_ms < target_ms:
        if shot_idx >= len(shots):
            shot_idx = 0  # wrap around

        shot = shots[shot_idx]
        shot_idx += 1

        remaining_ms = target_ms - accumulated_ms

        # Try to fill `remaining_ms` from this shot
        clip = _cut_clip(shot, remaining_ms)
        if clip is None:
            raise ValueError(
                f"cannot fill {remaining_ms}ms from {shot.shot_id} "
                f"({MIN_CLIP_MS}–{MAX_CLIP_MS}ms, speed {MIN_SPEED}–{MAX_SPEED}x)"
            )

        clips.append(clip)
        accumulated_ms += clip.playback_duration_ms()

    # Verify frame-exact duration
    frames = _total_frames(clips, fps)
    # frames = target_ms * fps.num / (fps.den * 1000)
    # so: frames * fps.den * 1000 == target_ms * fps.num
    if frames * fps.den * 1000 != target_ms * fps.num:
        raise ValueError(
            f"duration {target_ms}ms is not frame-exact at {fps.num}/{fps.den} fps "
            f"(got {frames} frames = {frames * fps.den * 1000 / fps.num:.3f}ms)"
        )

    return clips


def _cut_clip(shot: Shot, target_ms: TimeMs) -> Clip | None:
    """Try to cut a clip from a shot to fill `target_ms` at playback.

    Returns the clip if successful, None if impossible.
    Searches for a source duration in [MIN_CLIP_MS, min(available, MAX_CLIP_MS)]
    and adjusts playback speed to reach the target duration.
    """
    max_available = min(shot.available_ms, MAX_CLIP_MS)
    if max_available < MIN_CLIP_MS:
        return None

    # Try source durations from longest to shortest, with fine granularity
    step = 1 if max_available - MIN_CLIP_MS < 200 else 50
    for source_ms in range(max_available, MIN_CLIP_MS - 1, -step):
        needed_speed = source_ms / target_ms
        if MIN_SPEED <= needed_speed <= MAX_SPEED:
            # Found a valid clip
            start_offset = (shot.available_ms - source_ms) // 2
            return Clip(
                shot_id=shot.shot_id,
                src_in_ms=start_offset,
                src_out_ms=start_offset + source_ms,
                speed=needed_speed,
            )

    # Fine search: try every millisecond if we haven't found one yet
    for source_ms in range(max_available, MIN_CLIP_MS - 1, -1):
        needed_speed = source_ms / target_ms
        if MIN_SPEED <= needed_speed <= MAX_SPEED:
            start_offset = (shot.available_ms - source_ms) // 2
            return Clip(
                shot_id=shot.shot_id,
                src_in_ms=start_offset,
                src_out_ms=start_offset + source_ms,
                speed=needed_speed,
            )

    return None


def _total_frames(clips: list[Clip], fps: Rational) -> FrameCount:
    """Sum playback durations in frames (frame-exact)."""
    total_ms = sum(c.playback_duration_ms() for c in clips)
    # frames = ms * (fps.num / fps.den / 1000)
    #        = ms * fps.num / (fps.den * 1000)
    # Compute as integer to avoid float precision loss
    frames = (total_ms * fps.num) // (fps.den * 1000)
    return frames
