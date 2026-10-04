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


class LlmCallRow(SQLModel, table=True):
    """One chat request to a provider (retries inside it are counted, not listed)."""

    __tablename__ = "llm_calls"

    id: str = Field(primary_key=True)
    job_id: str | None = Field(default=None, index=True)
    task: str = Field(index=True)
    provider: str
    model: str
    prompt_version: str
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
