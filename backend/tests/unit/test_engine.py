from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from offscreen.domain.artifact import ArtifactFile, Manifest
from offscreen.engine import (
    ArtifactRef,
    ArtifactStore,
    DependencyCycle,
    Engine,
    Scope,
    Stage,
    StageCanceled,
    StageContext,
    StageOutput,
    UnknownStage,
    compute_cache_key,
)


class Source(Stage):
    """No inputs; writes `text` (a param) to out.txt."""

    name = "t.source"
    version = 1
    lane = "cpu"

    def __init__(self, text: str = "hello") -> None:
        self.text = text
        self.runs = 0

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"text": self.text, "asset": scope["asset_id"]}

    def run(self, ctx: StageContext) -> StageOutput:
        self.runs += 1
        (ctx.out_dir / "out.txt").write_text(self.text)
        return StageOutput(meta={"n": len(self.text)})


class Upper(Stage):
    name = "t.upper"
    version = 1
    lane = "cpu"

    def __init__(self) -> None:
        self.runs = 0

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("t.source", scope)]

    def run(self, ctx: StageContext) -> StageOutput:
        self.runs += 1
        text = ctx.input("t.source").path("out.txt").read_text()
        (ctx.out_dir / "sub").mkdir()
        (ctx.out_dir / "sub" / "upper.txt").write_text(text.upper())
        return StageOutput()


SCOPE = {"asset_id": "ast_demo"}


@pytest.fixture
def store(tmp_path: Any) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def test_miss_runs_then_hit_skips(store: ArtifactStore) -> None:
    src, up = Source(), Upper()
    engine = Engine(store, [src, up])

    first = engine.ensure("t.upper", SCOPE)
    assert first.path("sub/upper.txt").read_text() == "HELLO"
    assert (src.runs, up.runs) == (1, 1)

    second = Engine(store, [src, up]).ensure("t.upper", SCOPE)
    assert (src.runs, up.runs) == (1, 1)
    assert second.cache_key == first.cache_key
    assert second.dir == first.dir


def test_on_resolved_reports_each_stage_upstream_first_with_cache_status(
    store: ArtifactStore,
) -> None:
    seen: list[tuple[str, bool]] = []
    stages = [Source(), Upper()]
    Engine(store, stages, on_resolved=lambda n, hit: seen.append((n, hit))).ensure("t.upper", SCOPE)
    assert seen == [("t.source", False), ("t.upper", False)]
    seen.clear()
    Engine(store, stages, on_resolved=lambda n, hit: seen.append((n, hit))).ensure("t.upper", SCOPE)
    assert seen == [("t.source", True), ("t.upper", True)]


class Resumable(Stage):
    """Does its work in pieces kept in the work directory; fails on a chosen piece."""

    name = "t.resumable"
    version = 1
    lane = "cpu"

    def __init__(self, fail_at: int | None) -> None:
        self.fail_at = fail_at
        self.computed: list[int] = []

    def run(self, ctx: StageContext) -> StageOutput:
        parts = []
        for i in range(4):
            piece = ctx.work_dir / f"{i}.txt"
            if not piece.exists():
                if i == self.fail_at:
                    raise RuntimeError("quota exhausted")
                self.computed.append(i)
                piece.write_text(str(i * i))
            parts.append(piece.read_text())
        (ctx.out_dir / "out.txt").write_text(",".join(parts))
        return StageOutput()


def test_work_dir_keeps_finished_pieces_across_a_failed_run(store: ArtifactStore) -> None:
    first = Resumable(fail_at=2)
    with pytest.raises(RuntimeError):
        Engine(store, [first]).ensure("t.resumable", SCOPE)
    assert first.computed == [0, 1]
    assert not list((store.root / "t.resumable").glob(".tmp-*"))  # staging is gone, work is not

    second = Resumable(fail_at=None)
    art = Engine(store, [second]).ensure("t.resumable", SCOPE)
    assert second.computed == [2, 3]  # only what was missing
    assert art.path("out.txt").read_text() == "0,1,4,9"
    assert not (store.root / ".work" / "t.resumable").exists() or not any(
        (store.root / ".work" / "t.resumable").iterdir()
    )  # cleaned after success


def test_work_dir_is_not_shared_between_different_computations(store: ArtifactStore) -> None:
    class Param(Resumable):
        def __init__(self, tag: str, fail_at: int | None) -> None:
            super().__init__(fail_at)
            self.tag = tag

        def params(self, scope: Scope) -> dict[str, Any]:
            return {"tag": self.tag}

    with pytest.raises(RuntimeError):
        Engine(store, [Param("a", 2)]).ensure("t.resumable", SCOPE)
    other = Param("b", None)  # other parameters: other cache key: nothing to resume from
    Engine(store, [other]).ensure("t.resumable", SCOPE)
    assert other.computed == [0, 1, 2, 3]


def test_context_without_a_store_has_no_work_dir() -> None:
    ctx = StageContext(stage="x", scope={}, inputs=[], out_dir=Path("."))
    with pytest.raises(RuntimeError, match="work directory"):
        _ = ctx.work_dir


def test_peek_reports_what_is_cached_without_running_anything(store: ArtifactStore) -> None:
    src, up = Source(), Upper()
    engine = Engine(store, [src, up])
    assert engine.peek("t.source", SCOPE) is None
    assert engine.peek("t.upper", SCOPE) is None

    built = engine.ensure("t.source", SCOPE)  # only the upstream exists
    assert engine.peek("t.source", SCOPE) == built
    assert engine.peek("t.upper", SCOPE) is None
    assert (src.runs, up.runs) == (1, 0)

    full = engine.ensure("t.upper", SCOPE)
    assert engine.peek("t.upper", SCOPE) == full
    assert engine.peek("t.upper", {"asset_id": "ast_other"}) is None  # another scope

    changed = Engine(store, [Source("other text"), Upper()])  # new params: upstream key changes
    assert changed.peek("t.source", SCOPE) is None
    assert changed.peek("t.upper", SCOPE) is None
    assert (src.runs, up.runs) == (1, 1)


def test_peek_unknown_stage(store: ArtifactStore) -> None:
    with pytest.raises(UnknownStage):
        Engine(store, [Source()]).peek("nope", SCOPE)


def test_manifest_lists_files_with_hashes_and_meta(store: ArtifactStore) -> None:
    art = Engine(store, [Source(), Upper()]).ensure("t.source", SCOPE)
    assert [f.path for f in art.manifest.files] == ["out.txt"]
    assert art.manifest.files[0].size == 5
    assert art.manifest.files[0].hash.startswith("sha256:")
    assert art.meta == {"n": 5}
    assert art.manifest.scope == SCOPE
    assert (art.dir / "manifest.json").is_file()


def test_bumping_stage_version_invalidates_it_and_downstream(store: ArtifactStore) -> None:
    src, up = Source(), Upper()
    Engine(store, [src, up]).ensure("t.upper", SCOPE)

    src.version = 2
    src.text = "hello!"  # a version bump normally comes with a changed output
    Engine(store, [src, up]).ensure("t.upper", SCOPE)
    assert (src.runs, up.runs) == (2, 2)


def test_unchanged_upstream_output_keeps_downstream_cached(store: ArtifactStore) -> None:
    src, up = Source(), Upper()
    Engine(store, [src, up]).ensure("t.upper", SCOPE)

    src.version = 2  # upstream reruns, but writes byte-identical output
    Engine(store, [src, up]).ensure("t.upper", SCOPE)
    assert src.runs == 2
    assert up.runs == 1


def test_scope_reaches_the_key_only_through_params_and_inputs(store: ArtifactStore) -> None:
    src, up = Source("a"), Upper()
    Engine(store, [src, up]).ensure("t.upper", SCOPE)
    # Source puts the asset id in its params, so a new scope re-runs it; its output is
    # byte-identical, so the downstream stage shares the cached result.
    Engine(store, [src, up]).ensure("t.upper", {"asset_id": "ast_other"})
    assert (src.runs, up.runs) == (2, 1)

    src.text = "b"  # a changed param changes the output, which invalidates downstream
    Engine(store, [src, up]).ensure("t.upper", SCOPE)
    assert (src.runs, up.runs) == (3, 2)


def test_failed_stage_leaves_no_artifact_or_staging(store: ArtifactStore) -> None:
    class Boom(Source):
        def run(self, ctx: StageContext) -> StageOutput:
            (ctx.out_dir / "partial.txt").write_text("x")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        Engine(store, [Boom()]).ensure("t.source", SCOPE)
    assert [p.name for p in (store.root / "t.source").iterdir()] == []

    ok = Source()
    Engine(store, [ok]).ensure("t.source", SCOPE)
    assert ok.runs == 1


def test_damaged_artifact_is_rebuilt(store: ArtifactStore) -> None:
    src = Source()
    art = Engine(store, [src]).ensure("t.source", SCOPE)
    (art.dir / "out.txt").write_text("tampered, and longer")
    again = Engine(store, [src]).ensure("t.source", SCOPE)
    assert src.runs == 2
    assert again.path("out.txt").read_text() == "hello"


def test_unknown_stage_and_cycle(store: ArtifactStore) -> None:
    with pytest.raises(UnknownStage):
        Engine(store, []).ensure("nope", SCOPE)

    class A(Stage):
        name, version, lane = "t.a", 1, "cpu"

        def inputs(self, scope: Scope) -> list[ArtifactRef]:
            return [ArtifactRef("t.b", scope)]

        def run(self, ctx: StageContext) -> StageOutput:
            return StageOutput()

    class B(A):
        name = "t.b"

        def inputs(self, scope: Scope) -> list[ArtifactRef]:
            return [ArtifactRef("t.a", scope)]

    with pytest.raises(DependencyCycle, match=r"t\.a -> t\.b -> t\.a"):
        Engine(store, [A(), B()]).ensure("t.a", SCOPE)


def test_duplicate_stage_names_rejected(store: ArtifactStore) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Engine(store, [Source(), Source()])


def test_cancel_before_run_and_progress_reporting(store: ArtifactStore) -> None:
    src = Source()
    with pytest.raises(StageCanceled):
        Engine(store, [src], is_canceled=lambda: True).ensure("t.source", SCOPE)
    assert src.runs == 0

    events: list[tuple[str, float, str]] = []

    class Chatty(Source):
        def run(self, ctx: StageContext) -> StageOutput:
            ctx.progress(0.5, "half")
            ctx.progress(7, "clamped")
            return super().run(ctx)

    Engine(store, [Chatty()], progress=lambda s, f, m: events.append((s, f, m))).ensure(
        "t.source", SCOPE
    )
    assert events == [("t.source", 0.5, "half"), ("t.source", 1.0, "clamped")]


def test_no_staging_dirs_survive_a_successful_run(store: ArtifactStore) -> None:
    Engine(store, [Source(), Upper()]).ensure("t.upper", SCOPE)
    names = [p.name for d in store.root.iterdir() for p in d.iterdir()]
    assert not [n for n in names if n.startswith(".tmp-")]


# --- cache key -------------------------------------------------------------------------

_json = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=8),
    lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=4), c, max_size=3),
    max_leaves=8,
)


@given(params=st.dictionaries(st.text(min_size=1, max_size=4), _json, max_size=5))
def test_cache_key_ignores_dict_order(params: dict[str, Any]) -> None:
    reordered = dict(reversed(list(params.items())))
    a = compute_cache_key("s", 1, ["sha256:aa"], params, {})
    b = compute_cache_key("s", 1, ["sha256:aa"], reordered, {})
    assert a == b


def test_cache_key_depends_on_every_component() -> None:
    base = compute_cache_key("s", 1, ["sha256:aa"], {"k": 1}, {"model": "m"})
    assert base.startswith("sha256:") and len(base) == 7 + 64
    variants = [
        compute_cache_key("s2", 1, ["sha256:aa"], {"k": 1}, {"model": "m"}),
        compute_cache_key("s", 2, ["sha256:aa"], {"k": 1}, {"model": "m"}),
        compute_cache_key("s", 1, ["sha256:bb"], {"k": 1}, {"model": "m"}),
        compute_cache_key("s", 1, ["sha256:aa"], {"k": 2}, {"model": "m"}),
        compute_cache_key("s", 1, ["sha256:aa"], {"k": 1}, {"model": "m2"}),
    ]
    assert len({base, *variants}) == 6


# --- manifest --------------------------------------------------------------------------

H = "sha256:" + "a" * 64


@pytest.mark.parametrize("bad", ["", "/abs", "../up", "a/../b", "a//b", "a\\b", "./a"])
def test_artifact_file_rejects_unclean_paths(bad: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactFile(path=bad, size=0, hash=H)


def test_manifest_files_must_be_sorted_and_unique() -> None:
    f = lambda p: ArtifactFile(path=p, size=1, hash=H)  # noqa: E731
    base: dict[str, Any] = {"stage": "s", "stage_version": 1, "cache_key": H}
    Manifest(**base, files=[f("a"), f("b")])
    with pytest.raises(ValidationError, match="sorted"):
        Manifest(**base, files=[f("b"), f("a")])
    with pytest.raises(ValidationError, match="sorted"):
        Manifest(**base, files=[f("a"), f("a")])


def test_content_hash_ignores_cache_key_and_meta_but_not_content() -> None:
    def make(key: str, meta: dict[str, Any], h: str) -> Manifest:
        return Manifest(
            stage="s",
            stage_version=1,
            cache_key=key,
            meta=meta,
            files=[ArtifactFile(path="a", size=1, hash=h)],
        )

    a = make("sha256:" + "1" * 64, {}, H)
    assert a.content_hash() == make("sha256:" + "2" * 64, {"x": 1}, H).content_hash()
    assert a.content_hash() != make("sha256:" + "1" * 64, {}, "sha256:" + "b" * 64).content_hash()
