"""File-level persistence helpers (IO lives here, never in domain/)."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from offscreen.domain.common import canonical_json

M = TypeVar("M", bound=BaseModel)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via temp file + fsync + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_model(path: Path, model: BaseModel) -> str:
    """Write a model as canonical JSON; returns its content hash ('sha256:...')."""
    data = canonical_json(model).encode("utf-8")
    atomic_write_bytes(path, data)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def read_model(path: Path, cls: type[M]) -> M:
    return cls.model_validate_json(path.read_bytes())


FINGERPRINT_SPAN = 16 * 1024 * 1024


def fingerprint_file(path: Path, span: int = FINGERPRINT_SPAN) -> str:
    """Cheap identity for huge media files: sha256 over the size plus the first and last
    `span` bytes (the whole file when it is no longer than 2 * span). Survives moves and
    renames; changes if the file is re-encoded or truncated."""
    size = path.stat().st_size
    h = hashlib.sha256(f"{size}:".encode())
    with path.open("rb") as f:
        if size <= 2 * span:
            h.update(f.read())
        else:
            h.update(f.read(span))
            f.seek(size - span)
            h.update(f.read(span))
    return "sha256:" + h.hexdigest()
