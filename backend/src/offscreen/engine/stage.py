"""Stage: one pure transformation from input artifacts to an output artifact."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from offscreen.domain.job import Lane
from offscreen.engine.artifacts import Artifact

Scope = dict[str, str]
"""What a stage run is about, e.g. `{"asset_id": "ast_…"}`."""


@dataclass(frozen=True)
class ArtifactRef:
    """A dependency: the output of `stage` for `scope`."""

    stage: str
    scope: Scope = field(default_factory=dict)


@dataclass(frozen=True)
class StageOutput:
    meta: dict[str, Any] = field(default_factory=dict)
    """Small JSON-serializable summary stored in the manifest (counts, durations…)."""


class StageCanceled(Exception):
    """Raised by the engine when cancellation was requested."""


@dataclass
class StageContext:
    stage: str
    scope: Scope
    inputs: Sequence[Artifact]
    """Upstream artifacts, in the order `Stage.inputs` declared them."""
    out_dir: Path
    """Private staging directory; everything the stage writes here becomes its artifact."""
    _progress: Callable[[str, float, str], None] = lambda _s, _f, _m: None
    _is_canceled: Callable[[], bool] = lambda: False
    _work_dir: Callable[[], Path] | None = None

    def progress(self, frac: float, msg: str = "") -> None:
        self._progress(self.stage, min(1.0, max(0.0, frac)), msg)

    def is_canceled(self) -> bool:
        return self._is_canceled()

    @property
    def work_dir(self) -> Path:
        """Scratch directory that outlives a failed run of this exact stage computation (same
        inputs, parameters, provider): finished pieces of long work can be kept there and reused
        by the next attempt. Removed once the stage succeeds; not part of the artifact."""
        if self._work_dir is None:
            raise RuntimeError("this context has no work directory")
        return self._work_dir()

    def input(self, stage: str) -> Artifact:
        """The single input artifact produced by `stage`."""
        found = [a for a in self.inputs if a.stage == stage]
        if len(found) != 1:
            raise LookupError(
                f"{self.stage}: expected exactly one input from {stage!r}, got {len(found)}"
            )
        return found[0]


class Stage(ABC):
    """Subclass, set `name`/`version`/`lane`, declare `inputs`, implement `run`.

    Bump `version` whenever the output's meaning changes; that invalidates the stage and
    everything downstream of it. Anything that is not an upstream artifact but still
    affects the output must appear in `params` (including values derived from `scope`,
    since the cache key does not include the scope itself) or `provider_info`.
    """

    name: str
    version: int
    lane: Lane

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return []

    def params(self, scope: Scope) -> dict[str, Any]:
        """Normalized, JSON-serializable parameters that affect the output."""
        return {}

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        """Provider id + model name + prompt version, for stages that call a model."""
        return {}

    @abstractmethod
    def run(self, ctx: StageContext) -> StageOutput:
        """Read `ctx.inputs`, write files into `ctx.out_dir`. Must be deterministic given
        its inputs, params and provider (modulo model nondeterminism)."""
