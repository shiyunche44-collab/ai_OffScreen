"""Ground truth people mark by hand (`data/annotations/<asset_id>/`): kept apart from every
analysis output, so re-running an analysis never touches it."""

from __future__ import annotations

from pathlib import Path

from offscreen.domain.index import CutAnnotations, SelectionAnnotations
from offscreen.store.files import read_model, write_model

CUTS_FILE = "cuts.json"
SELECTION_FILE = "selection.json"


class AnnotationsStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "annotations"

    def _path(self, asset_id: str, name: str) -> Path:
        if "/" in asset_id or "\\" in asset_id or asset_id in ("", ".", ".."):
            raise ValueError(f"bad asset id {asset_id!r}")
        return self.root / asset_id / name

    def read_cuts(self, asset_id: str) -> CutAnnotations | None:
        """The marked cuts, or None if nobody has marked this movie."""
        path = self._path(asset_id, CUTS_FILE)
        return read_model(path, CutAnnotations) if path.is_file() else None

    def write_cuts(self, doc: CutAnnotations) -> None:
        write_model(self._path(doc.asset_id, CUTS_FILE), doc)

    def read_selection(self, asset_id: str) -> SelectionAnnotations | None:
        """The labelled footage choices, or None if nobody has labelled this movie."""
        path = self._path(asset_id, SELECTION_FILE)
        return read_model(path, SelectionAnnotations) if path.is_file() else None

    def write_selection(self, doc: SelectionAnnotations) -> None:
        write_model(self._path(doc.asset_id, SELECTION_FILE), doc)
