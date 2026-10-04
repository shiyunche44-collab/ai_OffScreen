"""Provider ports: interfaces the stages depend on. Vendor SDKs live in adapters/ only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from offscreen.domain.common import TimeRange
from offscreen.domain.index import Word


class DetectionCanceled(RuntimeError):
    """Raised by a detector when `should_cancel` turned true mid-run."""


class ShotDetector(Protocol):
    """Finds hard cuts in a video."""

    @property
    def id(self) -> str:
        """Identifies the algorithm and its settings; part of the stage's cache key."""
        ...

    def detect(
        self,
        video: Path,
        *,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[TimeRange]:
        """Consecutive shots in source order, as raw detector output. They need not cover
        the whole video; `algo.shots.normalize_shots` makes them a clean partition.
        Raises DetectionCanceled if `should_cancel` returns true."""
        ...


class AsrCanceled(RuntimeError):
    """Raised by an ASR provider when `should_cancel` turned true mid-run."""


@dataclass(frozen=True)
class AsrSegment:
    """One recognized utterance. Ids are assigned later, by the stage."""

    start_ms: int
    end_ms: int
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass(frozen=True)
class AsrResult:
    language: str
    segments: list[AsrSegment]


class ASR(Protocol):
    @property
    def id(self) -> str:
        """Engine, model and settings; part of the stage's cache key."""
        ...

    def transcribe(
        self,
        wav: Path,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AsrResult:
        """Recognize speech in a 16 kHz mono wav, with word timestamps when the engine has
        them. Raises AsrCanceled if `should_cancel` returns true."""
        ...
