"""Stage artifacts: immutable, content-addressed output directories (ARCHITECTURE §6.2)."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import Field, field_validator, model_validator

from offscreen.domain.common import Sha256, Strict, Versioned


class ArtifactFile(Strict):
    path: str  # posix, relative to the artifact directory
    size: int = Field(ge=0)
    hash: Sha256

    @field_validator("path")
    @classmethod
    def _relative_posix(cls, v: str) -> str:
        parts = v.split("/")
        if not v or v.startswith("/") or "\\" in v or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"artifact file path must be a clean relative posix path: {v!r}")
        return v


class Manifest(Versioned):
    """`manifest.json` inside every artifact directory. Its presence marks the artifact
    complete. Deliberately holds no timestamps, so the same inputs give the same bytes."""

    stage: str
    stage_version: int = Field(ge=1)
    cache_key: Sha256
    scope: dict[str, str] = {}
    files: list[ArtifactFile] = []
    meta: dict[str, Any] = {}

    @model_validator(mode="after")
    def _sorted_unique(self) -> Manifest:
        paths = [f.path for f in self.files]
        if paths != sorted(set(paths)):
            raise ValueError("manifest files must be unique and sorted by path")
        return self

    def content_hash(self) -> str:
        """Hash of the output files only (paths + content). Downstream cache keys use this,
        not `cache_key`, so a re-run that yields identical output keeps downstream cached."""
        h = hashlib.sha256()
        for f in self.files:
            h.update(f"{f.path}\t{f.hash}\n".encode())
        return "sha256:" + h.hexdigest()
