"""The background-music library: a folder of music files plus tags (mood, tempo).

Files live in `data/library/bgm/` (sub-folders allowed); a track is registered by naming its
file. Plans refer to a track by its file, and the renderer resolves it in this folder."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from offscreen.config import AppConfig
from offscreen.domain.bgm import BgmLibrary, BgmTrack, Tempo
from offscreen.media.probe import ProbeError, probe_audio_duration_ms
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.store.bgm import BgmStore

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _slug(stem: str) -> str:
    out = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem).strip("_")
    return out or "track"


def _moods(moods: Sequence[str] | None) -> list[str]:
    seen: list[str] = []
    for m in moods or []:
        m = m.strip().lower()
        if m and m not in seen:
            seen.append(m)
    return seen


class BgmService:
    def __init__(self, cfg: AppConfig) -> None:
        self.store = BgmStore(cfg.data_dir)

    def list(self, *, mood: str | None = None, tempo: Tempo | None = None) -> list[BgmTrack]:
        found = self.store.read().tracks
        if mood:
            found = [t for t in found if mood.strip().lower() in t.moods]
        if tempo:
            found = [t for t in found if t.tempo == tempo]
        return list(found)

    def get(self, track_id: str) -> BgmTrack:
        found = next((t for t in self.store.read().tracks if t.id == track_id), None)
        if found is None:
            raise NotFound(f"track {track_id} is not in the music library")
        return found

    def path(self, track: BgmTrack) -> Path:
        return self.store.dir / track.file

    def add(
        self,
        file: str,
        *,
        track_id: str | None = None,
        name: str | None = None,
        moods: Sequence[str] | None = None,
        tempo: Tempo | None = None,
    ) -> BgmTrack:
        """Register `file` (relative to the library folder), or change the tags of a registered
        one. A file that changed since it was added is measured again."""
        library = self.store.read()
        rel = Path(file)
        if rel.is_absolute() or ".." in rel.parts:
            raise InvalidInput("name the file relative to the music folder")
        path = self.store.dir / rel
        if not path.is_file():
            raise InvalidInput(f"no file {file} in {self.store.dir}")
        old = next((t for t in library.tracks if t.file == rel.as_posix()), None)
        try:
            duration = probe_audio_duration_ms(path)
        except ProbeError as e:
            raise InvalidInput(f"{file} is not a usable sound file: {e}") from e
        new_id = track_id or (old.id if old else self._free_id(library, _slug(rel.stem)))
        clash = next((t for t in library.tracks if t.id == new_id and t is not old), None)
        if clash is not None:
            raise InvalidInput(f"the id {new_id} is taken by {clash.file}")
        try:
            track = BgmTrack(
                id=new_id,
                name=name or (old.name if old else rel.stem),
                file=rel.as_posix(),
                moods=_moods(moods) if moods is not None else (old.moods if old else []),
                tempo=tempo if tempo is not None else (old.tempo if old else None),
                duration_ms=duration,
                sha256=_digest(path),
            )
        except ValueError as e:
            raise InvalidInput(str(e)) from e
        rest = [t for t in library.tracks if t is not old]
        position = library.tracks.index(old) if old else len(rest)
        rest.insert(position, track)
        self.store.write(library.model_copy(update={"tracks": rest}))
        return track

    def scan(self) -> Sequence[BgmTrack]:
        """Register every music file in the folder that is not registered yet (untagged)."""
        known = {t.file for t in self.store.read().tracks}
        added: list[BgmTrack] = []
        for p in sorted(self.store.dir.rglob("*")) if self.store.dir.is_dir() else []:
            rel = p.relative_to(self.store.dir).as_posix()
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES and rel not in known:
                try:
                    added.append(self.add(rel))
                except InvalidInput:
                    continue  # not music after all
        return added

    def remove(self, track_id: str) -> None:
        """Take a track out of the library (the file stays in the folder)."""
        library = self.store.read()
        if all(t.id != track_id for t in library.tracks):
            raise NotFound(f"track {track_id} is not in the music library")
        self.store.write(
            library.model_copy(update={"tracks": [t for t in library.tracks if t.id != track_id]})
        )

    @staticmethod
    def _free_id(library: BgmLibrary, base: str) -> str:
        taken = {t.id for t in library.tracks}
        if base not in taken:
            return base
        n = 2
        while f"{base}_{n}" in taken:
            n += 1
        return f"{base}_{n}"
