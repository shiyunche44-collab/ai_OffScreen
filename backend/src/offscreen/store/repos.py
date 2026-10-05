"""Repositories over the SQLite tables. Callers deal in domain models, not rows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, update
from sqlmodel import col, select

from offscreen.domain.artifact import Manifest
from offscreen.domain.asset import MediaAsset
from offscreen.domain.common import canonical_json, new_id
from offscreen.domain.job import Job, JobStatus, Lane, can_transition
from offscreen.domain.llm import LlmCallRecord
from offscreen.domain.project import Project, ProjectOptions
from offscreen.store.db import Database
from offscreen.store.files import atomic_write_bytes
from offscreen.store.models import ArtifactRow, AssetRow, JobRow, LlmCallRow, ProjectRow, utcnow


class AssetRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, asset: MediaAsset) -> MediaAsset:
        """Register an asset. The same file (same fingerprint) yields the already-registered
        asset, so importing twice is a no-op."""
        with self.db.session() as s:
            existing = s.exec(
                select(AssetRow).where(AssetRow.fingerprint == asset.fingerprint)
            ).first()
            if existing is not None:
                return MediaAsset.model_validate_json(existing.probe_json)
            s.add(_asset_row(asset))
            return asset

    def get(self, asset_id: str) -> MediaAsset | None:
        with self.db.session() as s:
            row = s.get(AssetRow, asset_id)
            return MediaAsset.model_validate_json(row.probe_json) if row else None

    def find_by_fingerprint(self, fingerprint: str) -> MediaAsset | None:
        with self.db.session() as s:
            row = s.exec(select(AssetRow).where(AssetRow.fingerprint == fingerprint)).first()
            return MediaAsset.model_validate_json(row.probe_json) if row else None

    def list(self) -> list[MediaAsset]:
        with self.db.session() as s:
            rows = s.exec(select(AssetRow).order_by(col(AssetRow.created_at), AssetRow.id)).all()
            return [MediaAsset.model_validate_json(r.probe_json) for r in rows]

    def update(self, asset: MediaAsset) -> None:
        """Replace the stored document (e.g. after `derived` files were produced)."""
        with self.db.session() as s:
            row = s.get(AssetRow, asset.id)
            if row is None:
                raise KeyError(asset.id)
            row.title = asset.title
            row.source_path = asset.source_path
            row.fingerprint = asset.fingerprint
            row.probe_json = canonical_json(asset)
            s.add(row)


def _asset_row(asset: MediaAsset) -> AssetRow:
    return AssetRow(
        id=asset.id,
        title=asset.title,
        source_path=asset.source_path,
        fingerprint=asset.fingerprint,
        probe_json=canonical_json(asset),
    )


class ArtifactIndex:
    """Index of artifacts on disk, for lookup and LRU garbage collection. The files and
    their manifests remain the source of truth."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, manifest: Manifest, path: str) -> ArtifactRow:
        """Insert or refresh the row for a committed artifact; `path` is relative to data_dir."""
        with self.db.session() as s:
            row = s.get(ArtifactRow, manifest.cache_key) or ArtifactRow(
                cache_key=manifest.cache_key,
                stage=manifest.stage,
                stage_version=manifest.stage_version,
                scope="",
                path=path,
                content_hash="",
                size=0,
            )
            row.stage_version = manifest.stage_version
            row.scope = json.dumps(manifest.scope, sort_keys=True, ensure_ascii=False)
            row.path = path
            row.content_hash = manifest.content_hash()
            row.size = sum(f.size for f in manifest.files)
            row.last_used_at = utcnow()
            s.add(row)
            return row

    def get(self, cache_key: str) -> ArtifactRow | None:
        with self.db.session() as s:
            return s.get(ArtifactRow, cache_key)

    def touch(self, cache_key: str) -> None:
        """Mark as just used (called on cache hits)."""
        with self.db.session() as s:
            row = s.get(ArtifactRow, cache_key)
            if row is not None:
                row.last_used_at = utcnow()
                s.add(row)

    def list(self, stage: str | None = None) -> list[ArtifactRow]:
        with self.db.session() as s:
            q = select(ArtifactRow).order_by(col(ArtifactRow.last_used_at), ArtifactRow.cache_key)
            if stage is not None:
                q = q.where(ArtifactRow.stage == stage)
            return list(s.exec(q).all())

    def delete(self, cache_key: str) -> None:
        with self.db.session() as s:
            row = s.get(ArtifactRow, cache_key)
            if row is not None:
                s.delete(row)


class ProjectRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, asset_id: str, name: str, options: ProjectOptions) -> Project:
        row = ProjectRow(
            id=new_id("prj"), asset_id=asset_id, name=name, options_json=canonical_json(options)
        )
        with self.db.session() as s:
            s.add(row)
        return _project(row)

    def get(self, project_id: str) -> Project | None:
        with self.db.session() as s:
            row = s.get(ProjectRow, project_id)
            return _project(row) if row else None

    def list(self) -> list[Project]:
        """Newest first."""
        with self.db.session() as s:
            q = select(ProjectRow).order_by(
                col(ProjectRow.created_at).desc(), col(ProjectRow.id).desc()
            )
            return [_project(r) for r in s.exec(q).all()]


def _project(row: ProjectRow) -> Project:
    return Project(
        id=row.id,
        asset_id=row.asset_id,
        name=row.name,
        options=ProjectOptions.model_validate_json(row.options_json),
        created_at=row.created_at.replace(tzinfo=UTC),
    )


_JOBS = JobRow.__table__  # type: ignore[attr-defined]
_C = _JOBS.c  # Core columns: UPDATEs below are plain SQL statements, not ORM objects


class JobRepo:
    """The job queue. Every state change is one guarded UPDATE, so it is atomic across the API
    and worker processes sharing the SQLite file: two workers cannot claim the same job, and a
    run that was replaced (requeued after a missed heartbeat) cannot write its result.

    `now` parameters exist for tests; they are naive UTC like every datetime in the store."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def enqueue(
        self, stage: str, scope: Mapping[str, Any], lane: Lane, *, cache_key: str | None = None
    ) -> Job:
        job_id = new_id("job")
        row = JobRow(
            id=job_id,
            stage=stage,
            scope_json=json.dumps(dict(scope), sort_keys=True, ensure_ascii=False),
            lane=lane,
            status="queued",
            cache_key=cache_key,
            log_path=f"logs/{job_id}.log",
        )
        with self.db.session() as s:
            s.add(row)
        return _job(row)

    def get(self, job_id: str) -> Job | None:
        with self.db.session() as s:
            row = s.get(JobRow, job_id)
            return _job(row) if row else None

    def count(self, status: JobStatus) -> int:
        with self.db.session() as s:
            q = select(func.count()).select_from(JobRow).where(JobRow.status == status)
            return int(s.exec(q).one())

    def claim(self, lane: Lane, limit: int, *, now: datetime | None = None) -> Job | None:
        """Take the oldest claimable queued job of `lane`, unless `limit` jobs of that lane are
        already running (counted across all workers). One statement, hence atomic."""
        now = now or utcnow()
        oldest = (
            select(_C.id)
            .where(
                _C.lane == lane,
                _C.status == "queued",
                or_(_C.not_before.is_(None), _C.not_before <= now),
            )
            .order_by(_C.created_at, _C.id)
            .limit(1)
            .scalar_subquery()
        )
        running = (
            select(func.count()).select_from(_JOBS).where(_C.lane == lane, _C.status == "running")
        ).scalar_subquery()
        stmt = (
            update(_JOBS)
            .where(_C.id == oldest, running < limit)
            .values(
                status="running",
                attempt=_C.attempt + 1,
                started_at=now,
                heartbeat_at=now,
                finished_at=None,
                not_before=None,
                progress=0.0,
                message="",
            )
            .returning(_C.id)
        )
        with self.db.session() as s:
            job_id = s.connection().execute(stmt).scalar()
            row = s.get(JobRow, job_id) if job_id else None
            return _job(row) if row else None

    def heartbeat(self, job_id: str, attempt: int, *, now: datetime | None = None) -> bool | None:
        """Prove the run is alive. Returns whether cancellation was requested, or None when this
        run no longer owns the job (it was requeued, canceled or finished meanwhile)."""
        stmt = (
            update(_JOBS)
            .where(*_owned(job_id, attempt))
            .values(heartbeat_at=now or utcnow())
            .returning(_C.cancel_requested)
        )
        with self.db.session() as s:
            row = s.connection().execute(stmt).first()
            return None if row is None else bool(row[0])

    def report_progress(self, job_id: str, attempt: int, frac: float, message: str) -> None:
        stmt = (
            update(_JOBS)
            .where(*_owned(job_id, attempt))
            .values(progress=min(1.0, max(0.0, frac)), message=message)
        )
        with self.db.session() as s:
            s.connection().execute(stmt)

    def finish(
        self,
        job_id: str,
        attempt: int,
        status: JobStatus,
        *,
        error: str | None = None,
        message: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        """running -> succeeded | failed | canceled. False when the run no longer owns the job."""
        if status == "queued" or not can_transition("running", status):
            raise ValueError(f"a run cannot finish as {status!r}")
        values: dict[str, Any] = {"status": status, "finished_at": now or utcnow(), "error": error}
        if status == "succeeded":
            values["progress"] = 1.0
        if message is not None:
            values["message"] = message
        stmt = update(_JOBS).where(*_owned(job_id, attempt)).values(**values)
        with self.db.session() as s:
            return bool(s.connection().execute(stmt).rowcount)

    def reschedule(
        self,
        job_id: str,
        attempt: int,
        *,
        not_before: datetime | None,
        error: str | None,
        message: str,
    ) -> bool:
        """running -> queued (automatic retry with backoff, or a worker shutting down)."""
        stmt = (
            update(_JOBS)
            .where(*_owned(job_id, attempt))
            .values(status="queued", not_before=not_before, error=error, message=message)
        )
        with self.db.session() as s:
            return bool(s.connection().execute(stmt).rowcount)

    def request_cancel(self, job_id: str, *, now: datetime | None = None) -> Job | None:
        """queued -> canceled at once; running -> flagged, the worker notices on its next
        heartbeat and the stage stops at its next checkpoint. Finished jobs are left alone."""
        now = now or utcnow()
        with self.db.session() as s:
            conn = s.connection()
            conn.execute(
                update(_JOBS)
                .where(_C.id == job_id, _C.status == "queued")
                .values(status="canceled", finished_at=now, message="canceled", not_before=None)
            )
            conn.execute(
                update(_JOBS)
                .where(_C.id == job_id, _C.status == "running")
                .values(cancel_requested=True)
            )
            row = s.get(JobRow, job_id)
            if row is not None:
                s.refresh(row)
            return _job(row) if row else None

    def retry(self, job_id: str) -> Job:
        """failed -> queued, with a fresh attempt budget (manual retry)."""
        stmt = (
            update(_JOBS)
            .where(_C.id == job_id, _C.status == "failed")
            .values(
                status="queued",
                attempt=0,
                error=None,
                progress=0.0,
                message="",
                cancel_requested=False,
                not_before=None,
                started_at=None,
                finished_at=None,
            )
        )
        with self.db.session() as s:
            if not s.connection().execute(stmt).rowcount:
                row = s.get(JobRow, job_id)
                if row is None:
                    raise KeyError(job_id)
                raise ValueError(f"job {job_id} is {row.status}; only failed jobs can be retried")
            row = s.get(JobRow, job_id)
            assert row is not None
            s.refresh(row)
            return _job(row)

    def requeue_stale(self, timeout_s: float, *, now: datetime | None = None) -> list[str]:
        """Crash recovery: running jobs whose heartbeat is older than `timeout_s` go back to the
        queue (or to canceled, if cancellation had been requested). Returns the requeued ids."""
        now = now or utcnow()
        stale = (_C.status == "running", _C.heartbeat_at < now - timedelta(seconds=timeout_s))
        with self.db.session() as s:
            conn = s.connection()
            conn.execute(
                update(_JOBS)
                .where(*stale, _C.cancel_requested)
                .values(status="canceled", finished_at=now, message="canceled")
            )
            requeued = conn.execute(
                update(_JOBS)
                .where(*stale)
                .values(status="queued", message="requeued: worker stopped responding")
                .returning(_C.id)
            )
            return [r[0] for r in requeued.all()]

    def list(
        self, *, status: JobStatus | None = None, lane: Lane | None = None, limit: int = 200
    ) -> list[Job]:
        """Newest first."""
        with self.db.session() as s:
            q = select(JobRow).order_by(col(JobRow.created_at).desc(), col(JobRow.id).desc())
            if status is not None:
                q = q.where(JobRow.status == status)
            if lane is not None:
                q = q.where(JobRow.lane == lane)
            return [_job(r) for r in s.exec(q.limit(limit)).all()]


def _owned(job_id: str, attempt: int) -> tuple[Any, ...]:
    """The run `attempt` of `job_id` is the one currently executing."""
    return (_C.id == job_id, _C.status == "running", _C.attempt == attempt)


def _utc(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=UTC) if dt is not None else None


def _job(row: JobRow) -> Job:
    return Job(
        id=row.id,
        stage=row.stage,
        scope=json.loads(row.scope_json),
        lane=row.lane,
        status=row.status,
        progress=row.progress,
        message=row.message,
        cache_key=row.cache_key,
        attempt=row.attempt,
        error=row.error,
        log_path=row.log_path,
        created_at=row.created_at.replace(tzinfo=UTC),
        started_at=_utc(row.started_at),
        finished_at=_utc(row.finished_at),
    )


class LlmCallRepo:
    """`llm_calls` rows, with request and response bodies kept as files next to the other
    data (`root/llm_calls/<id>.req.json`), the database holding only the pointers."""

    def __init__(self, db: Database, root: Path) -> None:
        self.db = db
        self.root = root

    def add(self, rec: LlmCallRecord) -> LlmCallRow:
        call_id = new_id("llm")
        rel_dir = Path("llm_calls")
        (self.root / rel_dir).mkdir(parents=True, exist_ok=True)
        paths: dict[str, str | None] = {"req": None, "resp": None}
        for kind, body in (("req", rec.request), ("resp", rec.response)):
            if body:
                rel = rel_dir / f"{call_id}.{kind}.json"
                text = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
                atomic_write_bytes(self.root / rel, text.encode("utf-8"))
                paths[kind] = rel.as_posix()
        row = LlmCallRow(
            id=call_id,
            job_id=rec.job_id,
            task=rec.task,
            provider=rec.provider,
            model=rec.model,
            prompt_version=rec.prompt_version,
            status=rec.status,
            error=rec.error,
            retries=rec.retries,
            in_tokens=rec.in_tokens,
            out_tokens=rec.out_tokens,
            cached_tokens=rec.cached_tokens,
            cost_usd=rec.cost_usd,
            latency_ms=rec.latency_ms,
            req_path=paths["req"],
            resp_path=paths["resp"],
        )
        with self.db.session() as s:
            s.add(row)
        return row

    def list(self, *, task: str | None = None, job_id: str | None = None) -> list[LlmCallRow]:
        with self.db.session() as s:
            q = select(LlmCallRow).order_by(col(LlmCallRow.created_at), LlmCallRow.id)
            if task is not None:
                q = q.where(LlmCallRow.task == task)
            if job_id is not None:
                q = q.where(LlmCallRow.job_id == job_id)
            return list(s.exec(q).all())

    def totals(self) -> dict[str, int]:
        """Token totals over all calls, for the analysis cost report."""
        rows = self.list()
        return {
            "calls": len(rows),
            "in_tokens": sum(r.in_tokens for r in rows),
            "out_tokens": sum(r.out_tokens for r in rows),
            "cached_tokens": sum(r.cached_tokens for r in rows),
        }
