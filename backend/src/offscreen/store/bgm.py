"""The BGM library on disk: `data/library/bgm/bgm.json` next to the music files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from offscreen.domain.bgm import BgmLibrary
from offscreen.store.files import read_model, write_model

BGM_FILE = "bgm.json"


class BgmStore:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "library" / "bgm"

    @property
    def path(self) -> Path:
        return self.dir / BGM_FILE

    def read(self) -> BgmLibrary:
        """The library; empty until a track is added."""
        return read_model(self.path, BgmLibrary) if self.path.is_file() else BgmLibrary()

    def write(self, library: BgmLibrary) -> None:
        write_model(self.path, library)

    def digest(self) -> str:
        """Identity of the library's music for render cache keys: each file and its checksum."""
        lines = sorted(f"{t.file}:{t.sha256}" for t in self.read().tracks)
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
