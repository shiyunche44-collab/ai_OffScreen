"""Names and human edits for characters. Pure: no IO, no models.

`check_names` is the check the naming stage runs on the model's answers. The rest is the revision
layer (ARCHITECTURE §5.3): AI output stays as written, a person's edits are kept apart and applied
when reading (`apply_overrides`); because a re-clustering renumbers people, an edit carries the
face centre of the person it is about and `remap_overrides` finds them again by it."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np

from offscreen.domain.index import Character, CharacterOverride, FaceCluster

REMAP_SIMILARITY = 0.6
"""Two face centres this similar (cosine) are the same person across clusterings."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_names(
    replies: Mapping[str, tuple[str | None, str | None]],
    known_ids: Sequence[str],
    dialogue: Sequence[str],
) -> dict[str, list[str]]:
    """Problems per character id. `replies[id] = (name, evidence)`. An id the model skipped or
    invented, a name without evidence, and evidence that is not a line of the film (a quote may
    be part of a line) are problems."""
    text = " ".join(_norm(line) for line in dialogue)
    found: dict[str, list[str]] = {}
    for cid in known_ids:
        if cid not in replies:
            found.setdefault(cid, []).append("没有输出这个人物")
    for cid in replies:
        if cid not in known_ids:
            found.setdefault(cid, []).append("这个编号不在人物列表里")
    for cid, (name, evidence) in replies.items():
        if cid not in known_ids or not name:
            continue
        if not evidence or not evidence.strip():
            found.setdefault(cid, []).append(f"写了名字「{name}」，但没有给出台词依据 evidence")
        elif _norm(evidence) not in text:
            found.setdefault(cid, []).append(
                f"evidence「{evidence}」不是台词里的原句，请逐字摘自台词，或把 name 设为 null"
            )
    return found


def _unit(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def match_centres(
    wanted: Mapping[str, Sequence[float]],
    current: Mapping[str, Sequence[float]],
    threshold: float = REMAP_SIMILARITY,
) -> dict[str, str]:
    """`wanted` key -> `current` key for centres that are the same person, most similar pairs
    first, each current centre used at most once."""
    if not wanted or not current:
        return {}
    wk, ck = list(wanted), list(current)
    a = np.stack([_unit(wanted[k]) for k in wk])
    b = np.stack([_unit(current[k]) for k in ck])
    if a.shape[1] != b.shape[1]:
        return {}
    sims = a @ b.T
    pairs = sorted(
        ((float(sims[i, j]), i, j) for i in range(len(wk)) for j in range(len(ck))),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    out: dict[str, str] = {}
    used: set[int] = set()
    for sim, i, j in pairs:
        if sim < threshold:
            break
        if wk[i] in out or j in used:
            continue
        out[wk[i]] = ck[j]
        used.add(j)
    return out


def remap_overrides(
    overrides: Sequence[CharacterOverride], current: Mapping[str, Sequence[float]]
) -> tuple[list[CharacterOverride], list[CharacterOverride]]:
    """The edits written against an older clustering, re-addressed to the people of `current`
    (`id -> face centre`). Returns `(remapped, orphans)`: an edit whose person is gone is an
    orphan (kept in the file, applied to nobody). An edit without a centre keeps its id when
    that id exists."""
    with_centre = {i: o for i, o in enumerate(overrides) if o.centroid}
    where = match_centres({str(i): o.centroid or [] for i, o in with_centre.items()}, current)
    targets = match_centres(
        {str(i): o.merged_into_centroid for i, o in with_centre.items() if o.merged_into_centroid},
        current,
    )
    remapped: list[CharacterOverride] = []
    orphans: list[CharacterOverride] = []
    for i, o in enumerate(overrides):
        if o.centroid:
            new_id = where.get(str(i))
            if new_id is None:
                orphans.append(o)
                continue
            merged = targets.get(str(i)) if o.merged_into_centroid else o.merged_into
            remapped.append(o.model_copy(update={"character_id": new_id, "merged_into": merged}))
        elif o.character_id in current:
            remapped.append(o)
        else:
            orphans.append(o)
    latest = {o.character_id: o for o in remapped}  # two edits on one person: the later wins
    return list(latest.values()), orphans


def resolve_merges(overrides: Sequence[CharacterOverride]) -> dict[str, str]:
    """`source id -> final id` for merged characters, following chains; a cycle or a merge into
    a person who is ignored is not a merge."""
    direct = {o.character_id: o.merged_into for o in overrides if o.merged_into}
    ignored = {o.character_id for o in overrides if o.ignored}
    out: dict[str, str] = {}
    for src in direct:
        seen = {src}
        cur = direct[src]
        while cur in direct and cur not in seen:
            seen.add(cur)
            cur = direct[cur]
        if cur in seen or cur in ignored or cur == src:
            continue
        out[src] = cur
    return out


def apply_overrides(
    characters: Sequence[Character], overrides: Sequence[CharacterOverride]
) -> tuple[list[Character], dict[str, str], list[str]]:
    """The characters as people see them: `(characters, merged, ignored)`. Names and aliases a
    person set win over the AI's (`name_source` becomes "human"); ignored characters are left
    out; a merged one is left out and its face count added to the character it joined."""
    by_id = {o.character_id: o for o in overrides}
    ignored = sorted(o.character_id for o in overrides if o.ignored)
    merged = {s: t for s, t in resolve_merges(overrides).items() if s not in ignored}
    extra: dict[str, int] = {}
    for c in characters:
        target = merged.get(c.id)
        if target is not None and c.face_cluster is not None:
            extra[target] = extra.get(target, 0) + c.face_cluster.size
    out: list[Character] = []
    for c in characters:
        if c.id in ignored or c.id in merged:
            continue
        o = by_id.get(c.id)
        update: dict[str, object] = {}
        if o is not None and o.name is not None:
            update.update(name=o.name or None, name_source="human" if o.name else None)
        if o is not None and o.aliases is not None:
            update["aliases"] = list(o.aliases)
        if c.id in extra and c.face_cluster is not None:
            update["face_cluster"] = FaceCluster(
                size=c.face_cluster.size + extra[c.id],
                centroid_ref=c.face_cluster.centroid_ref,
                thumbnails=c.face_cluster.thumbnails,
            )
        out.append(c.model_copy(update=update) if update else c)
    return out, merged, ignored
