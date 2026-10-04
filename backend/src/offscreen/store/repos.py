"""Repositories over the SQLite tables. Callers deal in domain models, not rows."""

from __future__ import annotations

import json
from pathlib import Path

from sqlmodel import col, select

from offscreen.domain.artifact import Manifest
from offscreen.domain.asset import MediaAsset
from offscreen.domain.common import canonical_json, new_id
from offscreen.domain.llm import LlmCallRecord
from offscreen.store.db import Database
from offscreen.store.files import atomic_write_bytes
from offscreen.store.models import ArtifactRow, AssetRow, LlmCallRow, utcnow


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
