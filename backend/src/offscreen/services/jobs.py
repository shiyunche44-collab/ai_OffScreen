"""Background use cases shared by the API and the worker: analyze an asset, write the script,
build the plan, render the video.

Each use case enqueues one job whose `stage` is the target stage and whose scope carries the
asset and the run options. `execute` is what the worker runs for a job: it hands the same
`Pipeline` the CLI uses the job's progress and cancel hooks, so a stage behaves identically
whether a person types a command or presses a button.

A job pulls in whatever upstream stages are not cached yet (the engine's `ensure`), so asking for
a script before analysis simply analyzes first. The lane is that of the target stage; analysis is
the one exception: when it will run a local ASR model the job takes the gpu lane, so two such
jobs never share the GPU (ARCHITECTURE §3.3).
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any, Protocol

from offscreen.config import AppConfig
from offscreen.domain.asset import MediaAsset
from offscreen.domain.job import Job, JobCanceled, JobStatus, Lane
from offscreen.engine.stage import StageCanceled
from offscreen.media.ffmpeg import FFmpegCanceled
from offscreen.providers.ports import AsrCanceled, DetectionCanceled
from offscreen.services.errors import Conflict, InvalidInput, NotFound
from offscreen.services.pipeline import (
    DEFAULT_STYLE,
    Pipeline,
    Providers,
    RunOptions,
    build_providers,
)
from offscreen.stages.analysis.story import StoryStage
from offscreen.stages.creation.plan import PlanStage
from offscreen.stages.creation.script import ScriptStage
from offscreen.stages.output.render import RenderStage
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo, JobRepo

ANALYZE = StoryStage.name  # everything the script writer reads
GENERATE_SCRIPT = ScriptStage.name
BUILD_PLAN = PlanStage.name
RENDER = RenderStage.name

LOG_TAIL_BYTES = 256 * 1024

USE_CASE_STAGES = (ANALYZE, GENERATE_SCRIPT, BUILD_PLAN, RENDER)

_CANCEL_ERRORS = (StageCanceled, FFmpegCanceled, DetectionCanceled, AsrCanceled)


class JobRun(Protocol):
    """What `execute` needs from the worker's job context."""

    def progress(self, frac: float, msg: str = "") -> None: ...

    def is_canceled(self) -> bool: ...


def options_to_scope(opts: RunOptions) -> dict[str, Any]:
    return asdict(opts)


def options_from_scope(raw: dict[str, Any]) -> RunOptions:
    known = {k: raw[k] for k in ("minutes", "voice", "style", "spoil_ending") if k in raw}
    return RunOptions(**known)


class JobService:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        providers: Providers | None = None,
        db: Database | None = None,
    ) -> None:
        """`providers`: model adapters (tests pass fakes); built from the config on first use."""
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self._owns_db = db is None
        self.db = db or Database(cfg.data_dir / "offscreen.db")
        self.jobs = JobRepo(self.db)
        self.assets = AssetRepo(self.db)
        self._providers = providers
        self._lock = threading.Lock()
        self._submit_lock = threading.Lock()

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    @property
    def providers(self) -> Providers:
        with self._lock:
            if self._providers is None:
                self._providers = build_providers(self.cfg, self.db)
            return self._providers

    # ---- use cases: each returns the (new or already active) job -------------------------
    def analyze(self, asset_id: str) -> Job:
        """Shots, keyframes, transcript and story of an asset."""
        asset = self._asset(asset_id)
        local_asr = self.cfg.asr.provider == "faster_whisper" and not asset.subtitles_external
        lane: Lane = "gpu" if local_asr else StoryStage.lane
        return self._submit(ANALYZE, asset_id, RunOptions(), lane)

    def generate_script(self, asset_id: str, opts: RunOptions | None = None) -> Job:
        return self._submit(GENERATE_SCRIPT, asset_id, _checked(opts), ScriptStage.lane)

    def build_plan(self, asset_id: str, opts: RunOptions | None = None) -> Job:
        return self._submit(BUILD_PLAN, asset_id, _checked(opts), PlanStage.lane)

    def render(self, asset_id: str, opts: RunOptions | None = None) -> Job:
        return self._submit(RENDER, asset_id, _checked(opts), RenderStage.lane)

    # ---- job control ----------------------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self, *, status: JobStatus | None = None, lane: Lane | None = None) -> list[Job]:
        return self.jobs.list(status=status, lane=lane)

    def log(self, job_id: str) -> str:
        """The tail of the job's log file (empty until the job has started)."""
        job = self.jobs.get(job_id)
        if job is None:
            raise NotFound(f"unknown job {job_id}")
        if not job.log_path:
            return ""
        path = self.cfg.data_dir / job.log_path
        try:
            with path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - LOG_TAIL_BYTES))
                text = f.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        return text if size <= LOG_TAIL_BYTES else "… (earlier output cut)\n" + text

    def cancel(self, job_id: str) -> Job:
        job = self.jobs.request_cancel(job_id)
        if job is None:
            raise NotFound(f"unknown job {job_id}")
        return job

    def retry(self, job_id: str) -> Job:
        try:
            return self.jobs.retry(job_id)
        except KeyError as e:
            raise NotFound(f"unknown job {job_id}") from e
        except ValueError as e:
            raise Conflict(str(e)) from e

    # ---- what the worker runs ---------------------------------------------------------------
    def execute(self, job: Job, ctx: JobRun) -> None:
        """Run the job's target stage (and its missing upstream) to completion."""
        asset_id = str(job.scope["asset_id"])
        opts = options_from_scope(job.scope.get("options", {}))
        done = 0
        total = 1

        def progress(_stage: str, frac: float, msg: str) -> None:
            ctx.progress((done + frac) / total, f"{_stage} {msg}".strip())

        def resolved(stage: str, hit: bool) -> None:
            nonlocal done
            done += 1
            ctx.progress(done / total, f"{stage} {'cached' if hit else 'done'}")

        with Pipeline(
            self.cfg,
            self.providers,
            progress=progress,
            on_resolved=resolved,
            is_canceled=ctx.is_canceled,
            db=self.db,
        ) as pipeline:
            total = len(pipeline.stage_chain(job.stage, asset_id, opts))
            try:
                pipeline.run_stage(job.stage, asset_id, opts)
            except _CANCEL_ERRORS as e:
                raise JobCanceled(str(e)) from e

    # ---- internals --------------------------------------------------------------------------
    def _asset(self, asset_id: str) -> MediaAsset:
        asset = self.assets.get(asset_id)
        if asset is None:
            raise NotFound(f"unknown asset {asset_id}")
        return asset

    def _submit(self, stage: str, asset_id: str, opts: RunOptions, lane: Lane) -> Job:
        self._asset(asset_id)
        scope = {"asset_id": asset_id, "options": options_to_scope(opts)}
        with self._submit_lock:
            for status in ("queued", "running"):  # pressing a button twice must not queue twice
                for job in self.jobs.list(status=status):
                    if job.stage == stage and job.scope == scope:
                        return job
            return self.jobs.enqueue(stage, scope, lane)


def _checked(opts: RunOptions | None) -> RunOptions:
    opts = opts or RunOptions()
    if opts.minutes <= 0:
        raise InvalidInput("minutes must be positive")
    if not opts.style.strip():
        return RunOptions(opts.minutes, opts.voice, DEFAULT_STYLE, opts.spoil_ending)
    return opts
