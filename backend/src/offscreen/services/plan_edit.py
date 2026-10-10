"""Editing the plan: apply a person's operations to the current version, store the next one.

The plan is a versioned document like the script (`base_version` must be the current version).
What the operations need to know about the film (shot ranges, transcript lines) comes from the
analysis artifacts, which must have been built."""

from __future__ import annotations

from offscreen.algo.plan_build import same_content
from offscreen.algo.plan_edit import EditContext, PlanEditError, apply_edits
from offscreen.config import AppConfig
from offscreen.domain.index import Shots, Transcript
from offscreen.domain.plan import EditPlan
from offscreen.domain.plan_edit import PlanOp
from offscreen.services.errors import Conflict, InvalidInput, NotFound
from offscreen.services.jobs import JobService
from offscreen.services.pipeline import Pipeline
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.store.db import Database
from offscreen.store.documents import DocumentStore, StaleBase
from offscreen.store.repos import AssetRepo, ProjectRepo


class PlanEditService:
    def __init__(self, cfg: AppConfig, db: Database, jobs: JobService) -> None:
        self.cfg = cfg
        self.db = db
        self.jobs = jobs
        self.store = DocumentStore(db, cfg.data_dir)
        self.projects = ProjectRepo(db)
        self.assets = AssetRepo(db)

    def edit(self, project_id: str, ops: list[PlanOp], base_version: int) -> EditPlan:
        """Apply `ops` to version `base_version` of the plan, which must be the current one;
        stored as the next version (author human). Nothing is stored if the result is the plan
        as it was."""
        project = self.projects.get(project_id)
        if project is None:
            raise NotFound(f"project {project_id} not found")
        head = self.store.read(project_id, "plan", EditPlan)
        if head is None:
            raise NotFound("the project has no plan yet: build it first")
        if head.version != base_version:
            raise Conflict(
                f"the plan was updated (base_version {base_version}, current {head.version}); "
                "reload it before editing"
            )
        asset = self.assets.get(project.asset_id)
        if asset is None:
            raise NotFound(f"unknown asset {project.asset_id}")
        ctx = self._context(
            project.asset_id, asset.duration_ms, any(o.op == "insert_original" for o in ops)
        )
        try:
            segments = apply_edits(head.segments, ops, ctx)
        except PlanEditError as e:
            raise InvalidInput(str(e)) from e
        edited = head.model_copy(update={"segments": segments})
        if same_content(edited, head):
            return head

        def build(doc_id: str, version: int, parent: int | None) -> EditPlan:
            return edited.model_copy(
                update={
                    "id": doc_id,
                    "version": version,
                    "parent_version": parent,
                    "author": "human",
                }
            )

        try:
            return self.store.append(project_id, "plan", base_version, build, author="human")
        except StaleBase as e:
            raise Conflict(
                f"the plan was updated (base_version {base_version}, current {e.current}); "
                "reload it before editing"
            ) from e

    def _context(self, asset_id: str, duration_ms: int, need_transcript: bool) -> EditContext:
        with Pipeline(self.cfg, self.jobs.providers, db=self.db) as p:
            shots_art = p.peek("analysis.shots", asset_id)
            transcript_art = p.peek("analysis.transcript", asset_id) if need_transcript else None
        if shots_art is None:
            raise NotFound("analysis.shots has not been built yet")
        if need_transcript and transcript_art is None:
            raise NotFound("analysis.transcript has not been built yet")
        shots = shots_art.read_model(SHOTS_FILE, Shots)
        return EditContext(
            asset_id=asset_id,
            shot_ranges={s.id: (s.start_ms, s.end_ms) for s in shots.shots},
            asset_duration_ms=duration_ms,
            transcript=transcript_art.read_model(TRANSCRIPT_FILE, Transcript)
            if transcript_art
            else None,
        )
