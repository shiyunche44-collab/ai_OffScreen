"""The versioned document store: immutable `v{n}.json` files plus an index in `documents`.

A version is never rewritten. Saving appends a new one and moves the project's head pointer,
and only if the caller's `base_version` is still the head (optimistic locking, ARCHITECTURE
§6.5); otherwise `StaleBase` carries the actual head so the caller can show a 409. The check
is one guarded UPDATE inside the transaction that also inserts the row and writes the file,
so two writers cannot both create version n + 1."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from sqlalchemy import update
from sqlmodel import col, select

from offscreen.domain.common import new_id
from offscreen.domain.document import DocKind, DocumentVersion
from offscreen.domain.script import Author
from offscreen.store.db import Database
from offscreen.store.files import read_model, write_model
from offscreen.store.models import DocumentRow, ProjectRow

D = TypeVar("D", bound=BaseModel)

_ID_PREFIX: dict[DocKind, str] = {"script": "scr", "plan": "pln"}
_HEAD = {"script": "current_script_version", "plan": "current_plan_version"}
_PROJECTS = ProjectRow.__table__  # type: ignore[attr-defined]


class StaleBase(Exception):
    """`base_version` is not the head any more (or the document does not exist yet)."""

    def __init__(self, base_version: int | None, current: int | None) -> None:
        super().__init__(f"base_version {base_version} but the current version is {current}")
        self.base_version = base_version
        self.current = current


class DocumentStore:
    def __init__(self, db: Database, data_dir: Path) -> None:
        self.db = db
        self.data_dir = data_dir

    def _path(self, project_id: str, kind: DocKind, version: int) -> str:
        return f"projects/{project_id}/docs/{kind}/v{version}.json"

    def dir_for(self, project_id: str, kind: DocKind) -> Path:
        """The directory holding every version of the document (and, for a plan, the audio
        files its versions refer to, which `AudioRef.file` gives relative to it)."""
        return self.data_dir / f"projects/{project_id}/docs/{kind}"

    def head(self, project_id: str, kind: DocKind) -> int | None:
        """The current version number; None if there is none yet. KeyError: unknown project."""
        with self.db.session() as s:
            row = s.get(ProjectRow, project_id)
            if row is None:
                raise KeyError(project_id)
            return getattr(row, _HEAD[kind])  # type: ignore[no-any-return]

    def append(
        self,
        project_id: str,
        kind: DocKind,
        base_version: int | None,
        build: Callable[[str, int, int | None], D],
        *,
        author: Author,
    ) -> D:
        """Create the version after `base_version` (None: the first one). `build(doc_id,
        version, parent_version)` returns the document to store. Raises StaleBase."""
        column = _HEAD[kind]
        with self.db.session() as s:
            if s.get(ProjectRow, project_id) is None:
                raise KeyError(project_id)
            version = (base_version or 0) + 1
            moved = s.connection().execute(
                update(_PROJECTS)
                .where(_PROJECTS.c.id == project_id, _PROJECTS.c[column].is_(base_version))
                .values({column: version})
            )
            if moved.rowcount == 0:
                raise StaleBase(base_version, getattr(s.get(ProjectRow, project_id), column))
            if base_version is None:
                doc_id = new_id(_ID_PREFIX[kind])
            else:
                doc_id = s.get(DocumentRow, (project_id, kind, base_version)).id  # type: ignore[union-attr]
            doc = build(doc_id, version, base_version)
            path = self._path(project_id, kind, version)
            write_model(self.data_dir / path, doc)
            s.add(
                DocumentRow(
                    id=doc.id,  # type: ignore[attr-defined]  # the document's own id
                    project_id=project_id,
                    kind=kind,
                    version=version,
                    parent_version=base_version,
                    author=author,
                    path=path,
                )
            )
        return doc

    def read(
        self, project_id: str, kind: DocKind, cls: type[D], version: int | None = None
    ) -> D | None:
        """A version (default: the head); None if it does not exist."""
        with self.db.session() as s:
            project = s.get(ProjectRow, project_id)
            if project is None:
                return None
            wanted = version if version is not None else getattr(project, _HEAD[kind])
            row = s.get(DocumentRow, (project_id, kind, wanted)) if wanted else None
        return read_model(self.data_dir / row.path, cls) if row else None

    def versions(self, project_id: str, kind: DocKind) -> list[DocumentVersion]:
        """The history, newest first."""
        with self.db.session() as s:
            q = (
                select(DocumentRow)
                .where(DocumentRow.project_id == project_id, DocumentRow.kind == kind)
                .order_by(col(DocumentRow.version).desc())
            )
            return [
                DocumentVersion(
                    kind=kind,
                    version=r.version,
                    parent_version=r.parent_version,
                    author=r.author,
                    created_at=r.created_at.replace(tzinfo=UTC),
                )
                for r in s.exec(q).all()
            ]
