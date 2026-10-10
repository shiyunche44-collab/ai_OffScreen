"""Ranking the shots of a film for a piece of narration (ARCHITECTURE §7.3, steps ① and ②).

Candidates are the shots of the scenes the text cites plus the best matches of a vector search
(`algo.candidates`); each is scored by the weighted factors of `algo.scoring`, with a
character-bigram text similarity (it needs no spaces to split words on). The plan build cuts
footage from the best of these; the selection evaluation (`offscreen select evaluate`) measures
how often the best ones are right. Both use this one ranking."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np

from offscreen.algo.candidates import candidates_for_segment
from offscreen.algo.scoring import ScoringWeights, ShotScore, score_candidates
from offscreen.algo.search import search_text
from offscreen.algo.textsim import coverage
from offscreen.algo.vectors import rank_by_similarity, search_signal
from offscreen.domain.index import (
    Cast,
    Characters,
    Scene,
    ShotCaption,
    ShotIndex,
    ShotIndexEntry,
    Shots,
)
from offscreen.domain.plan import Clip
from offscreen.domain.script import ScriptSegment
from offscreen.engine import Artifact
from offscreen.providers.ports import Embedder
from offscreen.stages.analysis.embeddings import (
    IMAGE_VECTORS_FILE,
    SHOT_INDEX_FILE,
    TEXT_VECTORS_FILE,
)

NEUTRAL_QUALITY = 0.5
"""Sharpness / brightness assumed for a shot without a measurement."""


class SelectionError(RuntimeError):
    pass


class VectorSearch:
    """A segment's text compared with every shot, by picture and by description."""

    def __init__(self, ids: list[str], columns: list[tuple[str, Embedder, Any]]) -> None:
        self.ids = ids
        self.columns = columns

    def __call__(self, text: str) -> tuple[list[str], dict[str, float]]:
        ranked: dict[str, list[tuple[str, float]]] = {}
        for name, embedder, matrix in self.columns:
            (vector,) = embedder.embed_texts([text])
            try:
                ranked[name] = rank_by_similarity(self.ids, matrix, vector)
            except ValueError as e:
                raise SelectionError(f"{name} vectors do not match the query embedder: {e}") from e
        return search_signal(ranked)


def load_vector_search(
    embeddings: Artifact, image_embedder: Embedder | None, text_embedder: Embedder | None
) -> VectorSearch | None:
    """The search over the `analysis.embeddings` artifact with the embedders that made it
    (None when neither is configured)."""
    if image_embedder is None and text_embedder is None:
        return None
    index = embeddings.read_model(SHOT_INDEX_FILE, ShotIndex)
    ids = [e.shot_id for e in index.shots]
    columns: list[tuple[str, Embedder, Any]] = []
    if image_embedder is not None and index.image_model is not None:
        columns.append(("image", image_embedder, np.load(embeddings.path(IMAGE_VECTORS_FILE))))
    if text_embedder is not None and index.text_model is not None:
        columns.append(("text", text_embedder, np.load(embeddings.path(TEXT_VECTORS_FILE))))
    return VectorSearch(ids, columns)


class ShotRanker:
    def __init__(
        self,
        *,
        shots: Shots,
        scenes: dict[str, Scene],
        captions: dict[str, ShotCaption],
        weights: ScoringWeights,
        search: VectorSearch | None,
    ) -> None:
        self.asset_id = shots.asset_id
        self.shot_list = shots
        self.scenes = scenes
        self.scene_shots = {sid: list(sc.shot_ids) for sid, sc in scenes.items()}
        self.captions = captions
        self.weights = weights
        self.search = search
        scene_of = {sh: sc.id for sc in scenes.values() for sh in sc.shot_ids}
        self.entries = {
            s.id: ShotIndexEntry(
                shot_id=s.id,
                scene_id=scene_of.get(s.id),
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                sharpness=s.quality.sharpness if s.quality else NEUTRAL_QUALITY,
                brightness=s.quality.brightness if s.quality else NEUTRAL_QUALITY,
                is_credits=bool(captions[s.id].is_credits) if s.id in captions else False,
                caption=search_text(captions[s.id]) if s.id in captions else "",
            )
            for s in shots.shots
        }

    def rank(
        self,
        seg: ScriptSegment,
        *,
        used: Collection[str] = (),
        taken: Sequence[Clip] = (),
    ) -> list[ShotScore]:
        """The usable candidates for `seg`, best first. `used` are shots shown already (they
        score lower), `taken` footage that is not available (original sound already plays it).
        Credits are never candidates."""
        for ref in seg.scene_refs:
            if ref not in self.scenes:
                raise SelectionError(f"{seg.id} cites unknown scene {ref}")
        top, similarity = self.search(seg.text) if self.search else ([], {})
        cand = candidates_for_segment(
            seg,
            self.shot_list,
            self.scene_shots,
            self.captions,
            Cast(asset_id=self.asset_id, shots=[]),
            Characters(asset_id=self.asset_id, characters=[]),
            {"embedding": top} if top else None,
        )
        ids = [sid for sid in cand.shot_ids if self._usable(sid, taken)]
        if not ids:
            raise SelectionError(
                f"{seg.id}: no usable footage among {len(cand.shot_ids)} candidates"
            )
        starts = [self.scenes[ref].start_ms for ref in seg.scene_refs]
        return score_candidates(
            ids,
            self.entries,
            {},
            self.captions,
            seg.text,
            similarity,
            set(),
            set(used),
            min(starts),
            self.weights,
            caption_similarity=coverage,
        )

    def _usable(self, shot_id: str, taken: Sequence[Clip]) -> bool:
        entry = self.entries.get(shot_id)
        if entry is None or entry.is_credits:
            return False
        return not any(t.src_in_ms < entry.end_ms and entry.start_ms < t.src_out_ms for t in taken)
