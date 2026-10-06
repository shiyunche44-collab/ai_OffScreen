"""Hand-marked ground truth and how the analysis measures up against it.

For shot detection (M3-01): a person marks the real cuts of a test film; `evaluate_cuts` reports
precision / recall / F1 of the detected shot boundaries (within a tolerance of a few frames)."""

from __future__ import annotations

from pydantic import BaseModel

from offscreen.algo.cuts import DEFAULT_TOLERANCE, evaluate_cuts, shot_cut_frames
from offscreen.config import AppConfig
from offscreen.domain.asset import MediaAsset
from offscreen.domain.index import CutAnnotations, Shots
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.jobs import JobService
from offscreen.services.pipeline import Pipeline
from offscreen.stages.analysis.proxy import PROXY_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
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
