"""Measuring footage selection against hand-labelled acceptable shots. Pure: rankings in,
numbers out.

For each labelled piece of narration the system produces a ranking of shots (best first). A
label lists every shot a person would accept under that text. Reported (IMPLEMENTATION_PLAN
M5-11):
- first-choice rate: how often the best-ranked shot is an acceptable one;
- top-k hit rate: how often at least one acceptable shot is among the first k;
- top-k recall: of the acceptable shots a label could have in the first k (at most k), the
  share that are there, averaged over labels;
- mean reciprocal rank of the first acceptable shot."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

DEFAULT_K = 5


@dataclass(frozen=True)
class LabelResult:
    label_id: str
    first_choice: str | None
    """The best-ranked shot (None when nothing was ranked)."""
    first_rank: int | None
    """Position (from 1) of the first acceptable shot; None when none was ranked at all."""
    top_k_hit: bool
    top_k_recall: float


@dataclass(frozen=True)
class SelectionScore:
    k: int
    labels: tuple[LabelResult, ...]
    first_choice_rate: float
    top_k_hit_rate: float
    top_k_recall: float
    mean_reciprocal_rank: float


def evaluate_selection(
    rankings: Mapping[str, Sequence[str]],
    acceptable: Mapping[str, Collection[str]],
    k: int = DEFAULT_K,
) -> SelectionScore:
    """Score `rankings` (label id -> shot ids, best first) against `acceptable` (label id ->
    the shots a person accepted). Every label in `acceptable` is scored; one without a ranking
    counts as a miss."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not acceptable:
        raise ValueError("nothing to evaluate: no labels")
    results: list[LabelResult] = []
    for label_id, good in acceptable.items():
        if not good:
            raise ValueError(f"{label_id}: no acceptable shots given")
        ranked = list(rankings.get(label_id, ()))
        first_rank = next((i for i, s in enumerate(ranked, start=1) if s in good), None)
        in_top = {s for s in ranked[:k] if s in good}
        results.append(
            LabelResult(
                label_id=label_id,
                first_choice=ranked[0] if ranked else None,
                first_rank=first_rank,
                top_k_hit=bool(in_top),
                top_k_recall=len(in_top) / min(len(set(good)), k),
            )
        )
    n = len(results)
    return SelectionScore(
        k=k,
        labels=tuple(results),
        first_choice_rate=sum(r.first_rank == 1 for r in results) / n,
        top_k_hit_rate=sum(r.top_k_hit for r in results) / n,
        top_k_recall=sum(r.top_k_recall for r in results) / n,
        mean_reciprocal_rank=sum(1 / r.first_rank for r in results if r.first_rank) / n,
    )
