"""Helpers for the story stage. Pure: no IO, no models.

Every statement a story makes about the plot cites scene ids (the anchor against hallucination,
ARCHITECTURE §5.3); `check_story_refs` is the check the stage runs on the model's answers."""

from __future__ import annotations

from collections.abc import Sequence


def fmt_clock(ms: int) -> str:
    s = ms // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def check_story_refs(
    act_scene_ids: Sequence[Sequence[str]],
    turning_point_ids: Sequence[str],
    valid_ids: Sequence[str],
    *,
    ending_ids: Sequence[str] = (),
    cover: bool = False,
) -> list[str]:
    """Problems with the scene ids a story cites: unknown ids, a scene in two acts, and (with
    `cover`) scenes that no act contains. Turning points and the ending's ids must exist too."""
    valid = set(valid_ids)
    errors: list[str] = []
    seen: dict[str, int] = {}
    for n, ids in enumerate(act_scene_ids, 1):
        for sid in ids:
            if sid not in valid:
                errors.append(f"第 {n} 幕引用了不存在的场景 {sid}")
            elif sid in seen:
                errors.append(f"场景 {sid} 同时出现在第 {seen[sid]} 幕和第 {n} 幕")
            else:
                seen[sid] = n
    if cover:
        lost = [sid for sid in valid_ids if sid not in seen]
        if lost:
            shown = "、".join(lost[:8]) + ("……" if len(lost) > 8 else "")
            errors.append(f"这些场景不属于任何一幕：{shown}")
    errors.extend(
        f"关键转折引用了不存在的场景 {sid}" for sid in turning_point_ids if sid not in valid
    )
    errors.extend(f"结局引用了不存在的场景 {sid}" for sid in ending_ids if sid not in valid)
    return errors
