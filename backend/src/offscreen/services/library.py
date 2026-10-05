"""The library: imported assets, projects, and what has been built for each.

Read-mostly use cases for the API. "Built" is answered from the artifact cache without running
anything, so these calls are cheap and never need a model key."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from offscreen.config import AppConfig
from offscreen.domain.asset import MediaAsset
from offscreen.domain.job import Job
from offscreen.domain.project import Project, ProjectOptions
from offscreen.media.probe import ProbeError
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.jobs import ANALYZE, RENDER, JobService
from offscreen.services.pipeline import Pipeline, RunOptions
from offscreen.stages.analysis.ingest import IngestError, ingest
from offscreen.stages.output.render import FINAL_FILE
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo, ProjectRepo


class StageStatus(BaseModel):
    stage: str
    cached: bool
    """The stage's output is on disk and up to date (so is everything upstream)."""


class AssetDetail(BaseModel):
    asset: MediaAsset
    stages: list[StageStatus]
    """What analysis builds (the chain behind its target stage), upstream first."""


class ProjectDetail(BaseModel):
    project: Project
    stages: list[StageStatus]
    """The whole chain up to the rendered video for this project's options, upstream first."""
    video: str | None
    """The finished video, relative to the data directory (None until rendered)."""


def run_options(opts: ProjectOptions) -> RunOptions:
    return RunOptions(opts.minutes, opts.voice, opts.style, opts.spoil_ending)


class LibraryService:
    def __init__(self, cfg: AppConfig, db: Database, jobs: JobService) -> None:
        self.cfg = cfg
        self.db = db
        self.jobs = jobs
        self.assets = AssetRepo(db)
        self.projects = ProjectRepo(db)

    # ---- assets ----------------------------------------------------------------------------
    def import_asset(self, path: str, title: str | None = None) -> MediaAsset:
        """Register a movie file that lives under one of the configured media roots. Importing
        the same file again returns the asset it already is."""
        file = self._inside_media_roots(path)
        try:
            return ingest(file, self.assets, title=title).asset
        except (IngestError, ProbeError) as e:
            raise InvalidInput(str(e)) from e

    def list_assets(self) -> list[MediaAsset]:
        return self.assets.list()

    def asset(self, asset_id: str) -> AssetDetail:
        asset = self._asset(asset_id)
        with self._pipeline() as p:
            names = p.stage_chain(ANALYZE, asset_id)
            stages = [StageStatus(stage=n, cached=p.peek(n, asset_id) is not None) for n in names]
        return AssetDetail(asset=asset, stages=stages)

    def analyze(self, asset_id: str) -> Job:
        return self.jobs.analyze(asset_id)

    # ---- projects --------------------------------------------------------------------------
    def create_project(
        self, asset_id: str, name: str | None = None, options: ProjectOptions | None = None
    ) -> Project:
        asset = self._asset(asset_id)
        label = (name or "").strip() or asset.title
        return self.projects.add(asset_id, label, options or ProjectOptions())

    def list_projects(self) -> list[Project]:
        return self.projects.list()

    def project(self, project_id: str) -> ProjectDetail:
        project = self._project(project_id)
        opts = run_options(project.options)
        with self._pipeline() as p:
            stages = [
                StageStatus(stage=n, cached=p.peek(n, project.asset_id, opts) is not None)
                for n in p.stage_chain(RENDER, project.asset_id, opts)
            ]
            final = p.peek(RENDER, project.asset_id, opts)
            video = (
                final.path(FINAL_FILE).resolve().relative_to(self.cfg.data_dir.resolve()).as_posix()
                if final is not None
                else None
            )
        return ProjectDetail(project=project, stages=stages, video=video)

    def generate_script(self, project_id: str) -> Job:
        p = self._project(project_id)
        return self.jobs.generate_script(p.asset_id, run_options(p.options))

    def build_plan(self, project_id: str) -> Job:
        p = self._project(project_id)
        return self.jobs.build_plan(p.asset_id, run_options(p.options))

    def render(self, project_id: str) -> Job:
        p = self._project(project_id)
        return self.jobs.render(p.asset_id, run_options(p.options))

    # ---- internals -------------------------------------------------------------------------
    def _pipeline(self) -> Pipeline:
        return Pipeline(self.cfg, self.jobs.providers, db=self.db)

    def _asset(self, asset_id: str) -> MediaAsset:
        asset = self.assets.get(asset_id)
        if asset is None:
            raise NotFound(f"unknown asset {asset_id}")
        return asset

    def _project(self, project_id: str) -> Project:
        project = self.projects.get(project_id)
        if project is None:
            raise NotFound(f"unknown project {project_id}")
        return project

    def _inside_media_roots(self, path: str) -> Path:
        roots = [r.expanduser().resolve() for r in self.cfg.media_roots]
        if not roots:
            raise InvalidInput("no media_roots configured; add the folder holding your movies")
        file = Path(path).expanduser().resolve()  # resolves symlinks and `..`
        if not any(file.is_relative_to(r) for r in roots):
            raise InvalidInput("the file is outside the configured media roots")
        if not file.is_file():
            raise InvalidInput(f"not a file: {path}")
        return file
