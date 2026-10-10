"""The voice library on disk: `data/library/voices/voices.json`, next to the reference
recordings voices are cloned from."""

from __future__ import annotations

from pathlib import Path

from offscreen.domain.voice import VoiceLibrary
from offscreen.store.files import read_model, write_model

VOICES_FILE = "voices.json"


class VoiceStore:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "library" / "voices"

    @property
    def path(self) -> Path:
        return self.dir / VOICES_FILE

    def read(self) -> VoiceLibrary:
        """The library; empty until a voice is added."""
        return read_model(self.path, VoiceLibrary) if self.path.is_file() else VoiceLibrary()

    def write(self, library: VoiceLibrary) -> None:
        write_model(self.path, library)
