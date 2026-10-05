"""Grouping faces by who they belong to (pure numpy, deterministic).

Face features are unit vectors, so similarity is the dot product (cosine). Thousands of faces are
first folded into tight micro-clusters in one pass over the input order (and one refinement pass
that moves each face to its nearest micro-cluster), then those are merged bottom-up by the
similarity of their centres until no pair is similar enough. Groups too small or seen in too few
shots are noise (extras, passers-by, false detections) and get no label.

The thresholds are for ArcFace features (same person roughly 0.5-0.8, different people < 0.35);
they have not been calibrated on film footage yet."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from offscreen.domain.index import FaceBox, ShotFaces

MICRO_SIMILARITY = 0.6
"""A face joins an existing micro-cluster at this similarity or above."""
MERGE_SIMILARITY = 0.4
"""Clusters whose centres are at least this similar are the same person."""
MIN_FACES = 3
MIN_SHOTS = 2
NOISE = -1

Vectors = NDArray[np.floating]


def _unit_rows(m: Vectors) -> Vectors:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def micro_clusters(vectors: Vectors, threshold: float = MICRO_SIMILARITY) -> NDArray[np.int64]:
    """A label per row: rows join the first-seen centre they are most similar to, if similar enough,
    else start a new one; then every row moves to its nearest final centre."""
    n = len(vectors)
    labels = np.zeros(n, dtype=np.int64)
    if n == 0:
        return labels
    sums = np.zeros((n, vectors.shape[1]), dtype=np.float64)  # at most n clusters
    k = 0
    for i, v in enumerate(vectors):
        if k:
            centres = _unit_rows(sums[:k])
            sims = centres @ v
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                labels[i] = best
                sums[best] += v
                continue
        labels[i] = k
        sums[k] = v
        k += 1
    centres = _unit_rows(sums[:k])
    nearest = np.argmax(vectors @ centres.T, axis=1)  # one refinement pass
    _, compact = np.unique(nearest, return_inverse=True)
    return compact.astype(np.int64)


def merge_clusters(
    vectors: Vectors, labels: NDArray[np.int64], threshold: float = MERGE_SIMILARITY
) -> NDArray[np.int64]:
    """Merge the most similar pair of cluster centres until none reaches `threshold`."""
    if len(vectors) == 0:
        return labels
    ids = list(np.unique(labels))
    sums = np.stack([vectors[labels == c].sum(axis=0) for c in ids])
    members = [[c] for c in ids]
    while len(members) > 1:
        centres = _unit_rows(sums)
        sims = centres @ centres.T
        np.fill_diagonal(sims, -np.inf)
        flat = int(np.argmax(sims))
        a, b = divmod(flat, len(members))
        if sims[a, b] < threshold:
            break
        a, b = min(a, b), max(a, b)
        sums[a] += sums[b]
        members[a] += members[b]
        sums = np.delete(sums, b, axis=0)
        del members[b]
    out = np.empty_like(labels)
    for new, group in enumerate(members):
        for old in group:
            out[labels == old] = new
    return out


def cluster_faces(
    vectors: Vectors,
    shot_of: Sequence[int],
    *,
    micro: float = MICRO_SIMILARITY,
    merge: float = MERGE_SIMILARITY,
    min_faces: int = MIN_FACES,
    min_shots: int = MIN_SHOTS,
) -> NDArray[np.int64]:
    """Person labels 0, 1, ... (largest group first) for each row of `vectors`, `NOISE` for
    faces that belong to no group big enough. `shot_of[i]` says which shot row i comes from."""
    if len(vectors) != len(shot_of):
        raise ValueError("one shot index per face is needed")
    if len(vectors) == 0:
        return np.zeros(0, dtype=np.int64)
    unit = _unit_rows(np.asarray(vectors, dtype=np.float64))
    labels = merge_clusters(unit, micro_clusters(unit, micro), merge)
    shots = np.asarray(shot_of)
    keep = [
        c
        for c in np.unique(labels)
        if (labels == c).sum() >= min_faces and len(np.unique(shots[labels == c])) >= min_shots
    ]
    keep.sort(key=lambda c: (-int((labels == c).sum()), int(np.argmax(labels == c))))
    out = np.full(len(labels), NOISE, dtype=np.int64)
    for rank, c in enumerate(keep):
        out[labels == c] = rank
    return out


def centroids(vectors: Vectors, labels: NDArray[np.int64]) -> Vectors:
    """The unit-length mean of each labelled group, one row per label 0..max."""
    count = int(labels.max()) + 1 if len(labels) and labels.max() >= 0 else 0
    if count == 0:
        return np.zeros((0, vectors.shape[1] if vectors.ndim == 2 else 0))
    return _unit_rows(np.stack([vectors[labels == c].sum(axis=0) for c in range(count)]))


def shot_cast(faces: Sequence[FaceBox]) -> list[tuple[str, float, float]]:
    """Who is in a shot: `(character_id, share, largest area_ratio)` for every identified
    character, by share of the shot's identified face area (summing to 1), largest first."""
    area: dict[str, float] = {}
    largest: dict[str, float] = {}
    for f in faces:
        if f.character_id is None:
            continue
        area[f.character_id] = area.get(f.character_id, 0.0) + f.area_ratio
        largest[f.character_id] = max(largest.get(f.character_id, 0.0), f.area_ratio)
    total = sum(area.values())
    if total <= 0:
        return []
    ranked = sorted(area, key=lambda c: (-area[c], c))
    return [(c, area[c] / total, largest[c]) for c in ranked]


def shot_indexes(shots: Sequence[ShotFaces]) -> list[int]:
    """For each face in document order, the index of its shot."""
    return [i for i, s in enumerate(shots) for _ in s.faces]
