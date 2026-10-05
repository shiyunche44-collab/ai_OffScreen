"""Which files under the data directory may be served over HTTP.

The rule is deliberately small: a relative path that, after resolving symlinks, stays inside the
data directory and names a regular file. SQLite files are the one exception, because the database
holds everything and nothing in the UI needs it. Every refusal is the same `NotFound`, so a probe
learns nothing about what exists outside."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Literal

from offscreen.config import AppConfig
from offscreen.services.errors import NotFound

_DB_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal", ".sqlite", ".sqlite3")
IMMUTABLE_PREFIX = "artifacts/"
"""Artifact directories are named by their content hash, so their files never change."""


class FileService:
    def __init__(self, cfg: AppConfig) -> None:
        self.root = cfg.data_dir

    def resolve(self, rel: str) -> Path:
        """The file at `rel` (relative to the data directory), or `NotFound`."""
        if not rel or "\0" in rel or "\\" in rel:
            raise NotFound("no such file")
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts:
            raise NotFound("no such file")
        root = self.root.resolve()
        try:
            target = (root / pure).resolve()
        except (OSError, ValueError) as e:
            raise NotFound("no such file") from e
        if not target.is_relative_to(root) or not target.is_file():
            raise NotFound("no such file")
        if target.name.lower().endswith(_DB_SUFFIXES):
            raise NotFound("no such file")
        return target

    @staticmethod
    def is_immutable(rel: str) -> bool:
        return rel.startswith(IMMUTABLE_PREFIX)


_RANGE_SPEC = re.compile(r"^\s*(\d*)\s*-\s*(\d*)\s*$")


def check_range(header: str, size: int) -> Literal["ok", "malformed", "unsatisfiable"]:
    """Judge a `Range` header against a file of `size` bytes (RFC 9110 §14.1.2, bytes units).
    The response itself is built by the web framework; this only decides whether to refuse, so
    the refusal can use the API's error shape."""
    unit, eq, specs = header.partition("=")
    if unit.strip().lower() != "bytes" or not eq:
        return "malformed"
    satisfiable = False
    for spec in specs.split(","):
        m = _RANGE_SPEC.match(spec)
        if m is None or (not m[1] and not m[2]):
            return "malformed"
        if not m[1]:  # suffix: the last n bytes
            satisfiable = satisfiable or (int(m[2]) > 0 and size > 0)
            continue
        start = int(m[1])
        if m[2] and int(m[2]) < start:
            return "malformed"
        satisfiable = satisfiable or start < size
    return "ok" if satisfiable else "unsatisfiable"
