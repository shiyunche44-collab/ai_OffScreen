"""Previewing one segment of the plan: a small, fast render from the proxy film.

Works on any stored version of the plan (default: the current one), so a person can listen to
and watch a segment right after editing it, without building anything else. Files go to
`previews/<hash>.mp4` in the data directory (served like the rest) and are reused for as long
as the segment, the film and the preview settings are the same."""

from __future__ import annotations

from pydantic import BaseModel, Field

from offscreen.config import AppConfig
from offscreen.domain.plan import EditPlan
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.jobs import JobService
from offscreen.services.pipeline import Pipeline
from offscreen.stages.analysis.proxy import AUDIO_48K, PROXY_FILE
from offscreen.stages.output.preview import PreviewError, render_segment_preview
from offscreen.store.db import Database
from offscreen.store.documents import DocumentStore
from offscreen.store.repos import AssetRepo, ProjectRepo

PREVIEWS_DIR = "previews"


class SegmentPreviewView(BaseModel):
    file: str = Field(description="Relative to the data directory; served at /api/files/….")
    cached: bool = Field(description="True when an earlier preview of the same segment was used.")
    segment_hash: str
    duration_ms: int
    plan_version: int


class PreviewService:
    def __init__(self, cfg: AppConfig, db: Database, jobs: JobService) -> None:
        self.cfg = cfg
        self.db = db
        self.jobs = jobs
        self.store = DocumentStore(db, cfg.data_dir)
        self.projects = ProjectRepo(db)
        self.assets = AssetRepo(db)

    def segment(
        self, project_id: str, segment_id: str, version: int | None = None
    ) -> SegmentPreviewView:
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
        if segment_id not in {s.id for s in plan.segments}:
            raise NotFound(f"the plan has no segment {segment_id}")
        asset = self.assets.get(project.asset_id)
        if asset is None:
            raise NotFound(f"unknown asset {project.asset_id}")
        with Pipeline(self.cfg, self.jobs.providers, db=self.db) as p:
            proxy = p.peek("analysis.proxy", project.asset_id)
        if proxy is None:
            raise NotFound("analysis.proxy has not been built yet")
        try:
            result = render_segment_preview(
                plan,
                segment_id,
                plan_dir=self.store.dir_for(project_id, "plan"),
                asset=asset,
                proxy=proxy.path(PROXY_FILE),
                proxy_hash=proxy.content_hash,
                audio=proxy.path(AUDIO_48K) if asset.audio else None,
                previews_dir=self.cfg.data_dir / PREVIEWS_DIR,
            )
        except PreviewError as e:
            raise InvalidInput(str(e)) from e
        return SegmentPreviewView(
            file=result.file.relative_to(self.cfg.data_dir).as_posix(),
            cached=result.cached,
            segment_hash=result.segment_hash,
            duration_ms=result.duration_ms,
            plan_version=plan.version,
        )
