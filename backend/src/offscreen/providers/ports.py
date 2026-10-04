"""Provider ports: interfaces the stages depend on. Vendor SDKs live in adapters/ only."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from offscreen.domain.common import TimeRange


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
