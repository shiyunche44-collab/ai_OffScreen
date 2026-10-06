"""SQLModel table definitions. The schema itself is owned by `migrations.py`; a test checks
these classes and the migrated database agree. Datetimes are naive UTC."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import NaiveDatetime
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AssetRow(SQLModel, table=True):
    __tablename__ = "assets"

    id: str = Field(primary_key=True)
    title: str
    source_path: str
    fingerprint: str = Field(unique=True, index=True)
    probe_json: str  # canonical JSON of the domain MediaAsset
    created_at: NaiveDatetime = Field(default_factory=utcnow)


class ArtifactRow(SQLModel, table=True):
    __tablename__ = "artifacts"

    cache_key: str = Field(primary_key=True)
    stage: str = Field(index=True)
    stage_version: int
    scope: str  # canonical JSON object
    path: str  # artifact directory, relative to data_dir
    content_hash: str
    size: int  # total bytes of the artifact's files
    created_at: NaiveDatetime = Field(default_factory=utcnow)
    last_used_at: NaiveDatetime = Field(default_factory=utcnow, index=True)


class ProjectRow(SQLModel, table=True):
    __tablename__ = "projects"

    id: str = Field(primary_key=True)
    asset_id: str = Field(foreign_key="assets.id", index=True)
    name: str
    options_json: str  # canonical JSON of the domain ProjectOptions
    created_at: NaiveDatetime = Field(default_factory=utcnow, index=True)
    current_script_version: int | None = None  # the versioned documents' head (see DocumentRow)
    current_plan_version: int | None = None


class DocumentRow(SQLModel, table=True):
    """One immutable version of a project's script or plan (ARCHITECTURE §5.4, §9.2). The
    content is the file at `path`; `id` is the document's own id (`scr_…`), shared by all its
    versions. The head version is `projects.current_<kind>_version`."""

    __tablename__ = "documents"

    id: str
    project_id: str = Field(primary_key=True, foreign_key="projects.id")
    kind: str = Field(primary_key=True)  # "script" | "plan"
    version: int = Field(primary_key=True)
    parent_version: int | None = None
    author: str  # "ai" | "human"
    path: str  # relative to data_dir
    created_at: NaiveDatetime = Field(default_factory=utcnow)


class JobRow(SQLModel, table=True):
    """A unit of background work (ARCHITECTURE §5.7). `attempt` counts runs since the last
    manual retry; a run is guarded by it, so a worker that lost its job cannot overwrite the
    run that replaced it."""

    __tablename__ = "jobs"

    id: str = Field(primary_key=True)
    stage: str = Field(index=True)
    scope_json: str  # canonical JSON object
    lane: str = Field(index=True)
    status: str = Field(index=True)
    progress: float = 0.0
    message: str = ""
    cache_key: str | None = None
    attempt: int = 0
    error: str | None = None
    log_path: str | None = None  # relative to data_dir
    cancel_requested: bool = False
    not_before: NaiveDatetime | None = None  # not claimed before this (retry backoff)
    heartbeat_at: NaiveDatetime | None = None
    created_at: NaiveDatetime = Field(default_factory=utcnow, index=True)
    started_at: NaiveDatetime | None = None
    finished_at: NaiveDatetime | None = None


class LlmCallRow(SQLModel, table=True):
    """One chat request to a provider (retries inside it are counted, not listed)."""

    __tablename__ = "llm_calls"

    id: str = Field(primary_key=True)
    job_id: str | None = Field(default=None, index=True)
    task: str = Field(index=True)
    provider: str
    model: str
    prompt_version: str
    stage: str | None = None  # the stage run that made the call, with its asset (None: outside one)
    asset_id: str | None = Field(default=None, index=True)
    status: str  # "ok" | "error"
    error: str | None = None
    retries: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0  # subscription plans count as 0; tokens are still recorded
    latency_ms: int = 0
    req_path: str | None = None  # request / response bodies, relative to data_dir
    resp_path: str | None = None
    created_at: NaiveDatetime = Field(default_factory=utcnow, index=True)


class StageRunRow(SQLModel, table=True):
    """One execution of a stage that was not served from the cache (a cache hit costs nothing and
    is not recorded). `asset_id` is the stage scope's asset, if it has one."""

    __tablename__ = "stage_runs"

    id: str = Field(primary_key=True)
    asset_id: str | None = Field(default=None, index=True)
    stage: str = Field(index=True)
    cache_key: str
    job_id: str | None = Field(default=None, index=True)
    status: str  # "ok" | "error" | "canceled"
    error: str | None = None
    duration_ms: int
    started_at: NaiveDatetime
