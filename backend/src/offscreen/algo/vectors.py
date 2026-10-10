"""Comparing a sentence with the shots by their vectors: the search signal of shot selection.

Pure; the vectors are what `analysis.embeddings` wrote (unit length, one row per shot). A query
is compared with every shot, which is a single matrix product even for a feature film."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from offscreen.algo.search import fuse

TOP_K = 20
"""How many shots the search signal adds to the candidates of a segment."""


def rank_by_similarity(
    shot_ids: Sequence[str], matrix: NDArray[np.float32], query: Sequence[float]
) -> list[tuple[str, float]]:
    """Every shot with its cosine similarity to `query`, best first (ties: shot order)."""
    if len(shot_ids) != matrix.shape[0]:
        raise ValueError(f"{len(shot_ids)} shots but {matrix.shape[0]} vectors")
    q = np.asarray(query, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("zero query vector")
    sims = matrix @ (q / norm)
    order = np.argsort(-sims, kind="stable")
    return [(shot_ids[i], float(sims[i])) for i in order]


def search_signal(
    columns: Mapping[str, list[tuple[str, float]]], top_k: int = TOP_K
) -> tuple[list[str], dict[str, float]]:
    """From one similarity ranking per vector column (image, text): the shots to add to the
    candidates (the best `top_k` of the rank fusion) and a score in [0, 1] for every shot.

    Cosine similarities of a picture model and a text model live on different scales (a good
    picture match is 0.3, a good text match 0.7), so each column is stretched to [0, 1] over
    its own range before the columns are averaged."""
    if not columns:
        return [], {}
    fused = fuse({name: [sid for sid, _ in ranked] for name, ranked in columns.items()})
    top = [sid for sid, _ in fused[:top_k]]
    stretched: list[dict[str, float]] = []
    for ranked in columns.values():
        if not ranked:
            continue
        sims = [s for _, s in ranked]
        lo, hi = min(sims), max(sims)
        span = hi - lo
        stretched.append({sid: (s - lo) / span if span > 0 else 0.0 for sid, s in ranked})
    scores = {
        sid: sum(col.get(sid, 0.0) for col in stretched) / len(stretched)
        for sid in {sid for col in stretched for sid in col}
    }
    return top, scores
