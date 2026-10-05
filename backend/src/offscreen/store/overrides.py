"""The human revision layer of the MovieIndex (ARCHITECTURE §5.3, §9.1): edits live in
`data/overrides/<asset_id>/`, apart from the AI output they correct, so a re-run of an analysis
never loses them."""

from __future__ import annotations

from pathlib import Path

from offscreen.domain.index import CharacterOverrides
from offscreen.store.files import read_model, write_model

CHARACTERS_OVERRIDES_FILE = "characters.overrides.json"


class OverridesStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "overrides"

    def _path(self, asset_id: str) -> Path:
        if "/" in asset_id or "\\" in asset_id or asset_id in ("", ".", ".."):
            raise ValueError(f"bad asset id {asset_id!r}")
        return self.root / asset_id / CHARACTERS_OVERRIDES_FILE

    def read_characters(self, asset_id: str) -> CharacterOverrides:
        """The edits made so far (none: an empty document)."""
        path = self._path(asset_id)
        if not path.is_file():
            return CharacterOverrides(asset_id=asset_id)
        return read_model(path, CharacterOverrides)

    def write_characters(self, doc: CharacterOverrides) -> None:
        write_model(self._path(doc.asset_id), doc)
