"""Pull-based executor: `ensure` builds whatever is missing, reuses whatever is cached."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from offscreen.engine.artifacts import Artifact, ArtifactStore
from offscreen.engine.cache import compute_cache_key
from offscreen.engine.stage import (
    ArtifactRef,
    Scope,
    Stage,
    StageCanceled,
    StageContext,
)

ProgressFn = Callable[[str, float, str], None]


class UnknownStage(KeyError):
    pass


class DependencyCycle(Exception):
    pass


def _memo_key(ref: ArtifactRef) -> tuple[str, tuple[tuple[str, str], ...]]:
    return ref.stage, tuple(sorted(ref.scope.items()))


class Engine:
    def __init__(
        self,
        store: ArtifactStore,
        stages: Iterable[Stage],
        *,
        progress: ProgressFn | None = None,
        is_canceled: Callable[[], bool] | None = None,
    ) -> None:
        self.store = store
        self.stages: dict[str, Stage] = {}
        for s in stages:
            if s.name in self.stages:
                raise ValueError(f"duplicate stage name: {s.name}")
            self.stages[s.name] = s
        self._progress = progress
        self._is_canceled = is_canceled

    def ensure(self, target: str, scope: Scope) -> Artifact:
        """The artifact of `target` for `scope`, running missing stages (upstream first)."""
        return self._ensure(ArtifactRef(target, dict(scope)), {}, [])

    def _ensure(
        self,
        ref: ArtifactRef,
        done: dict[tuple[str, tuple[tuple[str, str], ...]], Artifact],
        path: list[tuple[str, tuple[tuple[str, str], ...]]],
    ) -> Artifact:
        key = _memo_key(ref)
        if key in done:
            return done[key]
        if key in path:
            chain = " -> ".join(s for s, _ in [*path[path.index(key) :], key])
            raise DependencyCycle(chain)
        stage = self.stages.get(ref.stage)
        if stage is None:
            raise UnknownStage(ref.stage)

        path.append(key)
        upstream = [self._ensure(dep, done, path) for dep in stage.inputs(ref.scope)]
        path.pop()

        cache_key = compute_cache_key(
            stage.name,
            stage.version,
            [a.content_hash for a in upstream],
            stage.params(ref.scope),
            stage.provider_info(ref.scope),
        )
        artifact = self.store.get(stage.name, cache_key) or self._run(
            stage, ref.scope, upstream, cache_key
        )
        done[key] = artifact
        return artifact

    def _run(
        self, stage: Stage, scope: Scope, upstream: list[Artifact], cache_key: str
    ) -> Artifact:
        if self._is_canceled is not None and self._is_canceled():
            raise StageCanceled(stage.name)
        staging = self.store.begin(stage.name)
        try:
            ctx = StageContext(
                stage=stage.name,
                scope=scope,
                inputs=upstream,
                out_dir=staging,
                _progress=self._progress or (lambda _s, _f, _m: None),
                _is_canceled=self._is_canceled or (lambda: False),
            )
            output = stage.run(ctx)
            if ctx.is_canceled():
                raise StageCanceled(stage.name)
            return self.store.commit(
                staging,
                stage=stage.name,
                stage_version=stage.version,
                cache_key=cache_key,
                scope=scope,
                meta=output.meta,
            )
        except BaseException:
            self.store.discard(staging)
            raise
