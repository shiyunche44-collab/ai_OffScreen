"""ArtifactStore: content-addressed, immutable stage outputs on disk.

Layout: `{root}/{stage}/{cache_key hex}/` holding the stage's files plus `manifest.json`.
A stage writes into a private staging directory; `commit` hashes the files, writes the
manifest and renames the directory into place, so a directory that carries a manifest is
always complete.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from offscreen.domain.artifact import ArtifactFile, Manifest
from offscreen.store.files import read_model, write_model

MANIFEST = "manifest.json"
_CHUNK = 1 << 20

M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class Artifact:
    """A committed artifact: where it lives and what it contains."""

    dir: Path
    manifest: Manifest

    @property
    def stage(self) -> str:
        return self.manifest.stage

    @property
    def cache_key(self) -> str:
        return self.manifest.cache_key

    @property
    def content_hash(self) -> str:
        return self.manifest.content_hash()

    @property
    def meta(self) -> dict[str, Any]:
        return self.manifest.meta

    def path(self, rel: str) -> Path:
        """Absolute path of a file this artifact declares."""
        if rel not in {f.path for f in self.manifest.files}:
            raise FileNotFoundError(f"{self.stage}: artifact has no file {rel!r}")
        return self.dir / rel

    def read_model(self, rel: str, cls: type[M]) -> M:
        return read_model(self.path(rel), cls)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _hex(cache_key: str) -> str:
    return cache_key.removeprefix("sha256:")


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def dir_for(self, stage: str, cache_key: str) -> Path:
        return self.root / stage / _hex(cache_key)

    def get(self, stage: str, cache_key: str) -> Artifact | None:
        """The committed artifact for this key, or None on a miss. An artifact whose
        manifest is unreadable or whose files are missing/resized is treated as a miss."""
        d = self.dir_for(stage, cache_key)
        try:
            manifest = read_model(d / MANIFEST, Manifest)
        except (FileNotFoundError, ValidationError):
            return None
        if manifest.cache_key != cache_key or manifest.stage != stage:
            return None
        for f in manifest.files:
            p = d / f.path
            if not p.is_file() or p.stat().st_size != f.size:
                return None
        return Artifact(d, manifest)

    def begin(self, stage: str) -> Path:
        """A fresh, empty staging directory for a stage to write its output into."""
        staging = self.root / stage / f".tmp-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        return staging

    def discard(self, staging: Path) -> None:
        shutil.rmtree(staging, ignore_errors=True)

    def commit(
        self,
        staging: Path,
        *,
        stage: str,
        stage_version: int,
        cache_key: str,
        scope: dict[str, str],
        meta: dict[str, Any],
    ) -> Artifact:
        """Seal `staging` as the artifact for `cache_key` and return it."""
        files = [
            ArtifactFile(
                path=p.relative_to(staging).as_posix(), size=p.stat().st_size, hash=hash_file(p)
            )
            for p in sorted(staging.rglob("*"))
            if p.is_file()
        ]
        manifest = Manifest(
            stage=stage,
            stage_version=stage_version,
            cache_key=cache_key,
            scope=scope,
            files=sorted(files, key=lambda f: f.path),
            meta=meta,
        )
        write_model(staging / MANIFEST, manifest)
        final = self.dir_for(stage, cache_key)
        try:
            os.rename(staging, final)
        except OSError:
            # `final` exists: a concurrent run, or a corrupt leftover that `get` rejected.
            existing = self.get(stage, cache_key)
            if existing is not None:
                self.discard(staging)
                return existing
            shutil.rmtree(final, ignore_errors=True)
            os.rename(staging, final)
        return Artifact(final, manifest)
