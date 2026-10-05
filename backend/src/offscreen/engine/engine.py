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
ResolvedFn = Callable[[str, bool], None]
"""Called with `(stage name, cache hit)` once a stage's artifact is available, upstream first."""


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
        on_resolved: ResolvedFn | None = None,
    ) -> None:
        self.store = store
        self.stages: dict[str, Stage] = {}
        for s in stages:
            if s.name in self.stages:
                raise ValueError(f"duplicate stage name: {s.name}")
            self.stages[s.name] = s
        self._progress = progress
        self._is_canceled = is_canceled
        self._on_resolved = on_resolved

    def ensure(self, target: str, scope: Scope) -> Artifact:
        """The artifact of `target` for `scope`, running missing stages (upstream first)."""
        return self._ensure(ArtifactRef(target, dict(scope)), {}, [])

    def peek(self, target: str, scope: Scope) -> Artifact | None:
        """The cached artifact of `target` for `scope`, or None when it, or anything it depends
        on, is not built yet. Runs nothing (a status query)."""
        return self._peek(ArtifactRef(target, dict(scope)), {})

    def _peek(
        self,
        ref: ArtifactRef,
        memo: dict[tuple[str, tuple[tuple[str, str], ...]], Artifact | None],
    ) -> Artifact | None:
        key = _memo_key(ref)
        if key in memo:
            return memo[key]
        stage = self.stages.get(ref.stage)
        if stage is None:
            raise UnknownStage(ref.stage)
        upstream = [self._peek(dep, memo) for dep in stage.inputs(ref.scope)]
        found: Artifact | None = None
        if all(a is not None for a in upstream):
            cache_key = compute_cache_key(
                stage.name,
                stage.version,
                [a.content_hash for a in upstream if a is not None],
                stage.params(ref.scope),
                stage.provider_info(ref.scope),
            )
            found = self.store.get(stage.name, cache_key)
        memo[key] = found
        return found

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
        artifact = self.store.get(stage.name, cache_key)
        hit = artifact is not None
        if artifact is None:
            artifact = self._run(stage, ref.scope, upstream, cache_key)
        if self._on_resolved is not None:
            self._on_resolved(stage.name, hit)
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
                _work_dir=lambda: self.store.work_dir(stage.name, cache_key),
            )
            output = stage.run(ctx)
            if ctx.is_canceled():
                raise StageCanceled(stage.name)
            artifact = self.store.commit(
                staging,
                stage=stage.name,
                stage_version=stage.version,
                cache_key=cache_key,
                scope=scope,
                meta=output.meta,
            )
            self.store.clear_work(stage.name, cache_key)
            return artifact
        except BaseException:
            self.store.discard(staging)
            raise
