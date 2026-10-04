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
