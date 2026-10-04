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
