"""The background-music library: tracks in `library/bgm/`, tagged by mood and tempo."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from offscreen.domain.common import Strict, Versioned

Tempo = Literal["slow", "medium", "fast"]


class BgmTrack(Strict):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    """What a plan refers to (`Bgm.file` carries `file`, so a renamed id never breaks a plan)."""
    name: str = Field(min_length=1)
    file: str = Field(min_length=1)
    """Relative to `library/bgm/`."""
    moods: list[str] = []
    """Free tags such as "tense", "warm", "epic"; compared in lower case."""
    tempo: Tempo | None = None
    duration_ms: int = Field(gt=0)
    sha256: str = Field(min_length=64, max_length=64)
    """Of the file as it was added: a changed file is a different track to the render cache."""


class BgmLibrary(Versioned):
    tracks: list[BgmTrack] = []

    @model_validator(mode="after")
    def _unique(self) -> Self:
        for what, values in (
            ("ids", [t.id for t in self.tracks]),
            ("files", [t.file for t in self.tracks]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate track {what}")
        return self
