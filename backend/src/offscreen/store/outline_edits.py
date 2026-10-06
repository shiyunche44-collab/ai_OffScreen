"""A person's edit of a project's outline: `projects/<project_id>/outline.json`.

It sits beside the generated outline artifact, never inside it, so regenerating (a new style,
a new length) does not destroy it; the person decides whether to keep it or reset it."""

from __future__ import annotations

from pathlib import Path

from offscreen.domain.script import ScriptOutline
from offscreen.store.files import read_model, write_model

OUTLINE_EDIT_FILE = "outline.json"


class OutlineEditStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "projects"

    def _path(self, project_id: str) -> Path:
        if "/" in project_id or "\\" in project_id or project_id in ("", ".", ".."):
            raise ValueError(f"bad project id {project_id!r}")
        return self.root / project_id / OUTLINE_EDIT_FILE

    def read(self, project_id: str) -> ScriptOutline | None:
        path = self._path(project_id)
        return read_model(path, ScriptOutline) if path.is_file() else None

    def write(self, project_id: str, outline: ScriptOutline) -> None:
        write_model(self._path(project_id), outline)

    def delete(self, project_id: str) -> bool:
        path = self._path(project_id)
        existed = path.is_file()
        path.unlink(missing_ok=True)
        return existed
