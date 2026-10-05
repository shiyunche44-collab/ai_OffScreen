"""Pipeline engine: Stage base class, cache keys, ArtifactStore, `ensure()`."""

from offscreen.engine.artifacts import Artifact, ArtifactStore
from offscreen.engine.cache import compute_cache_key
from offscreen.engine.engine import DependencyCycle, Engine, StageRun, UnknownStage
from offscreen.engine.stage import (
    ArtifactRef,
    Scope,
    Stage,
    StageCanceled,
    StageContext,
    StageOutput,
)

__all__ = [
    "Artifact",
    "ArtifactRef",
    "ArtifactStore",
    "DependencyCycle",
    "Engine",
    "Scope",
    "Stage",
    "StageCanceled",
    "StageContext",
    "StageOutput",
    "StageRun",
    "UnknownStage",
    "compute_cache_key",
]
