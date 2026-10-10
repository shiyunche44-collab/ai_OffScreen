"""Footage candidates for one segment of the plan: what the plan editor's drawer offers.

The ranking is the one the plan build and the selection evaluation use (`ShotRanker`: the shots
of the cited scenes plus the best vector-search matches, weighted scores). Shots the rest of the
plan already shows rank lower (the reuse factor) and are named; footage that original sound plays
is left out."""

from __future__ import annotations

from pydantic import BaseModel, Field

from offscreen.algo.scoring import ScoringWeights
from offscreen.config import AppConfig
from offscreen.domain.index import Captions, Scenes, Shots
from offscreen.domain.plan import Clip, EditPlan
from offscreen.domain.script import Script
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.jobs import JobService
from offscreen.services.pipeline import Pipeline, Providers
from offscreen.stages.analysis.captions import CAPTIONS_FILE
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.creation.selection import (
    SelectionError,
    ShotRanker,
    VectorSearch,
    load_vector_search,
)
from offscreen.store.db import Database
from offscreen.store.documents import DocumentStore
from offscreen.store.repos import ProjectRepo

DEFAULT_LIMIT = 12
MAX_LIMIT = 50


def load_index(
    cfg: AppConfig, providers: Providers, db: Database, asset_id: str
) -> tuple[Shots, Scenes, Captions]:
    """Shots, scenes and captions of a movie; NotFound until they are built."""
    with Pipeline(cfg, providers, db=db) as p:
        arts = [
            p.peek(n, asset_id) for n in ("analysis.shots", "analysis.scenes", "analysis.captions")
        ]
    for name, art in zip(("shots", "scenes", "captions"), arts, strict=True):
        if art is None:
            raise NotFound(f"analysis.{name} has not been built yet")
    a, b, c = arts
    assert a is not None and b is not None and c is not None
    return (
        a.read_model(SHOTS_FILE, Shots),
        b.read_model(SCENES_FILE, Scenes),
        c.read_model(CAPTIONS_FILE, Captions),
    )


def load_search(
    cfg: AppConfig, providers: Providers, db: Database, asset_id: str
) -> VectorSearch | None:
    """The vector search when embedders are configured and their artifact is built."""
    if providers.image_embedder is None and providers.text_embedder is None:
        return None
    with Pipeline(cfg, providers, db=db) as p:
        art = p.peek("analysis.embeddings", asset_id)
    if art is None:
        return None
    return load_vector_search(art, providers.image_embedder, providers.text_embedder)


class ScoreView(BaseModel):
    """The factors of a score, each 0-1 (the total is their weighted sum)."""

    embedding: float
    caption: float
    character: float
    quality: float
    size: float
    reuse: float
    time_order: float
    exclusions: float


class CandidateView(BaseModel):
    shot_id: str
    scene_id: str | None
    start_ms: int
    end_ms: int
    score: float = Field(description="0-1; the candidates are listed best first.")
    factors: ScoreView
    caption: str
    used_by: str | None = Field(
        description="Another segment of the plan that shows this shot (it ranks lower for it)."
    )
    current: bool = Field(description="The segment shows this shot now.")


class CandidatesView(BaseModel):
    segment_id: str
    plan_version: int
    total: int = Field(description="How many usable candidates there are (the list is cut).")
    vector_search: bool
    candidates: list[CandidateView]


class SelectionService:
    def __init__(self, cfg: AppConfig, db: Database, jobs: JobService) -> None:
        self.cfg = cfg
        self.db = db
        self.jobs = jobs
        self.store = DocumentStore(db, cfg.data_dir)
        self.projects = ProjectRepo(db)

    def candidates(
        self,
        project_id: str,
        segment_id: str,
        version: int | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> CandidatesView:
        """The best footage for a narration segment of the plan, best first."""
        if not 1 <= limit <= MAX_LIMIT:
            raise InvalidInput(f"limit must be between 1 and {MAX_LIMIT}")
        project = self.projects.get(project_id)
        if project is None:
            raise NotFound(f"project {project_id} not found")
        plan = self.store.read(project_id, "plan", EditPlan, version)
        if plan is None:
            raise NotFound(
                "the project has no plan yet"
                if version is None
                else f"the plan has no version {version}"
            )
        seg = next((s for s in plan.segments if s.id == segment_id), None)
        if seg is None:
            raise NotFound(f"the plan has no segment {segment_id}")
        if seg.kind != "narration" or not seg.text:
            raise InvalidInput(f"{segment_id} is not a narration segment with text")
        script = self.store.read(project_id, "script", Script, plan.script_ref.version)
        source = next((s for s in script.segments if s.id == segment_id), None) if script else None
        if source is None or not source.scene_refs:
            raise InvalidInput(f"{segment_id} cites no scenes in the script the plan follows")

        providers = self.jobs.providers
        shots, scenes, captions = load_index(self.cfg, providers, self.db, project.asset_id)
        search = load_search(self.cfg, providers, self.db, project.asset_id)
        ranker = ShotRanker(
            shots=shots,
            scenes={s.id: s for s in scenes.scenes},
            captions={c.shot_id: c for c in captions.captions},
            weights=ScoringWeights(),
            search=search,
        )
        used_by: dict[str, str] = {}
        taken: list[Clip] = []
        for other in plan.segments:
            if other.id == seg.id:
                continue
            if other.kind == "original":
                taken.extend(other.clips)
            for clip in other.clips:
                if clip.shot_id:
                    used_by.setdefault(clip.shot_id, other.id)
        shown = {c.shot_id for c in seg.clips if c.shot_id}
        narration = source.model_copy(update={"text": seg.text})
        try:
            ranked = ranker.rank(narration, used=set(used_by), taken=taken)
        except SelectionError as e:
            raise InvalidInput(str(e)) from e
        return CandidatesView(
            segment_id=seg.id,
            plan_version=plan.version,
            total=len(ranked),
            vector_search=search is not None,
            candidates=[
                CandidateView(
                    shot_id=s.shot_id,
                    scene_id=ranker.entries[s.shot_id].scene_id,
                    start_ms=ranker.entries[s.shot_id].start_ms,
                    end_ms=ranker.entries[s.shot_id].end_ms,
                    score=round(s.total, 4),
                    factors=ScoreView(
                        embedding=round(s.embedding, 4),
                        caption=round(s.caption, 4),
                        character=round(s.character, 4),
                        quality=round(s.quality, 4),
                        size=round(s.size, 4),
                        reuse=round(s.reuse, 4),
                        time_order=round(s.time_order, 4),
                        exclusions=round(s.exclusions, 4),
                    ),
                    caption=ranker.entries[s.shot_id].caption,
                    used_by=used_by.get(s.shot_id),
                    current=s.shot_id in shown,
                )
                for s in ranked[:limit]
            ],
        )
