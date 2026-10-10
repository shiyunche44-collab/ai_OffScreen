"""Hand-marked ground truth and how the analysis measures up against it.

For shot detection (M3-01): a person marks the real cuts of a test film; `evaluate_cuts` reports
precision / recall / F1 of the detected shot boundaries (within a tolerance of a few frames).

For footage selection (M5-11): a person labels pieces of narration with the shots that would be
acceptable under them; `evaluate_selection` ranks the shots the way the plan build does and
reports how often the first choice is acceptable and how often one is among the top few."""

from __future__ import annotations

from pydantic import BaseModel, Field

from offscreen.algo.cuts import DEFAULT_TOLERANCE, evaluate_cuts, shot_cut_frames
from offscreen.algo.scoring import ScoringWeights
from offscreen.algo.selection_eval import DEFAULT_K, evaluate_selection
from offscreen.config import AppConfig
from offscreen.domain.asset import MediaAsset
from offscreen.domain.index import (
    Captions,
    CutAnnotations,
    Scenes,
    SelectionAnnotations,
    SelectionLabel,
    Shots,
)
from offscreen.domain.script import ScriptSegment
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.jobs import JobService
from offscreen.services.pipeline import Pipeline
from offscreen.services.selection import load_index, load_search
from offscreen.stages.analysis.proxy import PROXY_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.creation.selection import SelectionError, ShotRanker, VectorSearch
from offscreen.store.annotations import AnnotationsStore
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo


class CutsView(BaseModel):
    asset_id: str
    fps_num: int
    fps_den: int
    cuts: list[int]
    """Marked cut frames (empty until someone marks them)."""
    marked: bool
    """Anyone has saved marks for this movie."""


class CutEvaluation(BaseModel):
    asset_id: str
    source: str
    """What was measured: `shots` (the shot stage's boundaries, after merging / splitting) or
    `detector` (the detector's raw cuts)."""
    tolerance: int
    marked: int
    detected: int
    true_positives: int
    precision: float
    recall: float
    f1: float
    false_positives: list[int]
    false_negatives: list[int]


class LabelResultView(BaseModel):
    label_id: str
    text: str
    first_choice: str | None
    first_rank: int | None = Field(description="Where the first acceptable shot ranked (from 1).")
    top_k_hit: bool
    acceptable: list[str]


class SelectionEvaluation(BaseModel):
    asset_id: str
    labels: int
    k: int
    first_choice_rate: float
    top_k_hit_rate: float
    top_k_recall: float
    mean_reciprocal_rank: float
    vector_search: bool = Field(description="Whether the vector search was part of the ranking.")
    results: list[LabelResultView]
    """Every label, in the order given."""


class AnnotationService:
    def __init__(self, cfg: AppConfig, db: Database, jobs: JobService) -> None:
        self.cfg = cfg
        self.db = db
        self.jobs = jobs
        self.assets = AssetRepo(db)
        self.store = AnnotationsStore(cfg.data_dir)

    # ---- marks -----------------------------------------------------------------------------
    def cuts(self, asset_id: str) -> CutsView:
        asset = self._asset(asset_id)
        doc = self.store.read_cuts(asset_id)
        return self._view(asset, doc.cuts if doc else [], marked=doc is not None)

    def save_cuts(self, asset_id: str, cuts: list[int]) -> CutsView:
        """Replace the marked cuts (frames from 0). Order and duplicates are tidied; a frame
        beyond the end of the film is refused."""
        asset = self._asset(asset_id)
        frames = asset.video.fps.frames_for_ms(asset.duration_ms)
        tidy = sorted(set(cuts))
        if any(c < 1 for c in tidy):
            raise InvalidInput("a cut is the first frame of a new shot, so it is at least 1")
        if tidy and tidy[-1] >= frames:
            raise InvalidInput(f"frame {tidy[-1]} is past the end of the film ({frames} frames)")
        self.store.write_cuts(CutAnnotations(asset_id=asset_id, fps=asset.video.fps, cuts=tidy))
        return self._view(asset, tidy, marked=True)

    # ---- measuring -------------------------------------------------------------------------
    def evaluate_cuts(
        self,
        asset_id: str,
        *,
        tolerance: int = DEFAULT_TOLERANCE,
        source: str = "shots",
    ) -> CutEvaluation:
        asset = self._asset(asset_id)
        if tolerance < 0 or tolerance > 50:
            raise InvalidInput("tolerance must be between 0 and 50 frames")
        doc = self.store.read_cuts(asset_id)
        if doc is None:
            raise NotFound("no cuts have been marked for this movie yet")
        if doc.fps != asset.video.fps:
            raise InvalidInput("the marks were made at another frame rate than the movie's")
        detected = self._detected(asset, source)
        score = evaluate_cuts(detected, doc.cuts, tolerance)
        return CutEvaluation(
            asset_id=asset_id,
            source=source,
            tolerance=tolerance,
            marked=len(doc.cuts),
            detected=len(detected),
            true_positives=score.true_positives,
            precision=round(score.precision, 4),
            recall=round(score.recall, 4),
            f1=round(score.f1, 4),
            false_positives=list(score.false_positives),
            false_negatives=list(score.false_negatives),
        )

    def _detected(self, asset: MediaAsset, source: str) -> list[int]:
        with Pipeline(self.cfg, self.jobs.providers, db=self.db) as p:
            if source == "shots":
                art = p.peek("analysis.shots", asset.id)
                if art is None:
                    raise NotFound("analysis.shots has not been built yet")
                shots = art.read_model(SHOTS_FILE, Shots).shots
                return shot_cut_frames([s.start_ms for s in shots], asset.video.fps)
            if source == "detector":
                proxy = p.peek("analysis.proxy", asset.id)
                if proxy is None:
                    raise NotFound("analysis.proxy has not been built yet")
                raw = p.providers.detector.detect(proxy.path(PROXY_FILE))
                return shot_cut_frames([r.start_ms for r in raw], asset.video.fps)
        raise InvalidInput(f"unknown source {source!r}: use shots or detector")

    # ---- footage selection -----------------------------------------------------------------
    def selection(self, asset_id: str) -> list[SelectionLabel]:
        self._asset(asset_id)
        doc = self.store.read_selection(asset_id)
        return doc.labels if doc else []

    def save_selection(self, asset_id: str, labels: list[SelectionLabel]) -> list[SelectionLabel]:
        """Replace the labelled footage choices. Every scene and shot named must exist, so a
        typo is caught now and not as a silently missed label later."""
        self._asset(asset_id)
        shots, scenes, _ = self._index(asset_id)
        shot_ids = {s.id for s in shots.shots}
        scene_ids = {s.id for s in scenes.scenes}
        for label in labels:
            for ref in label.scene_refs:
                if ref not in scene_ids:
                    raise InvalidInput(f"{label.id}: unknown scene {ref}")
            for sid in label.acceptable:
                if sid not in shot_ids:
                    raise InvalidInput(f"{label.id}: unknown shot {sid}")
        try:
            doc = SelectionAnnotations(asset_id=asset_id, labels=labels)
        except ValueError as e:
            raise InvalidInput(str(e)) from e
        self.store.write_selection(doc)
        return doc.labels

    def evaluate_selection(self, asset_id: str, *, k: int = DEFAULT_K) -> SelectionEvaluation:
        """Rank the shots for every labelled text the way the plan build does (the same
        candidates, scores and vector search, with default weights) and measure the ranking."""
        self._asset(asset_id)
        if not 1 <= k <= 50:
            raise InvalidInput("k must be between 1 and 50")
        doc = self.store.read_selection(asset_id)
        if doc is None or not doc.labels:
            raise NotFound("no footage choices have been labelled for this movie yet")
        shots, scenes, captions = self._index(asset_id)
        search = self._search(asset_id)
        ranker = ShotRanker(
            shots=shots,
            scenes={s.id: s for s in scenes.scenes},
            captions={c.shot_id: c for c in captions.captions},
            weights=ScoringWeights(),
            search=search,
        )
        rankings: dict[str, list[str]] = {}
        for label in doc.labels:
            seg = ScriptSegment(
                id="seg_eval",
                kind="narration",
                text=label.text,
                scene_refs=label.scene_refs,
            )
            try:
                rankings[label.id] = [s.shot_id for s in ranker.rank(seg)]
            except SelectionError as e:
                raise InvalidInput(f"{label.id}: {e}") from e
        score = evaluate_selection(rankings, {x.id: set(x.acceptable) for x in doc.labels}, k)
        by_id = {x.id: x for x in doc.labels}
        return SelectionEvaluation(
            asset_id=asset_id,
            labels=len(doc.labels),
            k=k,
            first_choice_rate=round(score.first_choice_rate, 4),
            top_k_hit_rate=round(score.top_k_hit_rate, 4),
            top_k_recall=round(score.top_k_recall, 4),
            mean_reciprocal_rank=round(score.mean_reciprocal_rank, 4),
            vector_search=search is not None,
            results=[
                LabelResultView(
                    label_id=r.label_id,
                    text=by_id[r.label_id].text,
                    first_choice=r.first_choice,
                    first_rank=r.first_rank,
                    top_k_hit=r.top_k_hit,
                    acceptable=by_id[r.label_id].acceptable,
                )
                for r in score.labels
            ],
        )

    def _index(self, asset_id: str) -> tuple[Shots, Scenes, Captions]:
        return load_index(self.cfg, self.jobs.providers, self.db, asset_id)

    def _search(self, asset_id: str) -> VectorSearch | None:
        return load_search(self.cfg, self.jobs.providers, self.db, asset_id)

    # ---- internals -------------------------------------------------------------------------
    @staticmethod
    def _view(asset: MediaAsset, cuts: list[int], *, marked: bool) -> CutsView:
        fps = asset.video.fps
        return CutsView(
            asset_id=asset.id, fps_num=fps.num, fps_den=fps.den, cuts=cuts, marked=marked
        )

    def _asset(self, asset_id: str) -> MediaAsset:
        asset = self.assets.get(asset_id)
        if asset is None:
            raise NotFound(f"unknown asset {asset_id}")
        return asset
