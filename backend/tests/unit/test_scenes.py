"""The scenes stage: candidates, windowed decisions, summaries, resuming."""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from offscreen.domain.index import (
    Captions,
    Scenes,
    Shot,
    ShotCaption,
    Shots,
    ShotSignature,
    Transcript,
    TranscriptLine,
    VisualSignatures,
)
from offscreen.engine import ArtifactStore, Engine, Scope, Stage, StageContext, StageOutput
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.providers.ports import LLMQuotaExhausted, Message
from offscreen.stages.analysis.captions import CAPTIONS_FILE
from offscreen.stages.analysis.keyframes import SIGNATURES_FILE
from offscreen.stages.analysis.scenes import SCENES_FILE, ScenesError, ScenesStage
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.store.files import write_model

ASSET = "ast_t1"
SCOPE = {"asset_id": ASSET}


def hist(bin_: int) -> list[int]:
    h = [0] * 64
    h[bin_ % 64] = 1000
    return h


class StubKeyframes(Stage):
    """`n` shots of `shot_ms` each; `hist_of(i)` is the colour histogram of shot i's frames."""

    name = "analysis.keyframes"
    version = 1
    lane = "cpu"

    def __init__(self, n: int, shot_ms: int, hist_of: Any) -> None:
        self.n, self.shot_ms, self.hist_of = n, shot_ms, hist_of

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"n": self.n, "ms": self.shot_ms, "hists": [self.hist_of(i) for i in range(self.n)]}

    def run(self, ctx: StageContext) -> StageOutput:
        shots = [
            Shot(id=f"sh_{i:04d}", start_ms=i * self.shot_ms, end_ms=(i + 1) * self.shot_ms)
            for i in range(self.n)
        ]
        write_model(ctx.out_dir / SHOTS_FILE, Shots(asset_id=ASSET, shots=shots))
        sigs = [
            ShotSignature(shot_id=s.id, frames=[self.hist_of(i)] * 3) for i, s in enumerate(shots)
        ]
        write_model(
            ctx.out_dir / SIGNATURES_FILE,
            VisualSignatures(asset_id=ASSET, bins_per_channel=4, signatures=sigs),
        )
        return StageOutput()


class StubCaptions(Stage):
    """Every shot's caption starts with its place letter: `A·画面 3`."""

    name = "analysis.captions"
    version = 1
    lane = "api"

    def __init__(self, n: int, place_of: Any, drop: str | None = None) -> None:
        self.n, self.place_of, self.drop = n, place_of, drop

    def inputs(self, scope: Scope) -> list[Any]:
        from offscreen.engine import ArtifactRef

        return [ArtifactRef("analysis.keyframes", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"places": [self.place_of(i) for i in range(self.n)], "drop": self.drop}

    def run(self, ctx: StageContext) -> StageOutput:
        caps = [
            ShotCaption(
                shot_id=f"sh_{i:04d}", caption=f"{self.place_of(i)}·画面 {i}", is_credits=False
            )
            for i in range(self.n)
            if f"sh_{i:04d}" != self.drop
        ]
        write_model(ctx.out_dir / CAPTIONS_FILE, Captions(asset_id=ASSET, captions=caps))
        return StageOutput()


class StubTranscript(Stage):
    name = "analysis.transcript"
    version = 1
    lane = "cpu"

    def __init__(self, lines: list[tuple[int, int, str]]) -> None:
        self.lines = lines

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"lines": self.lines}

    def run(self, ctx: StageContext) -> StageOutput:
        doc = Transcript(
            asset_id=ASSET,
            language="en",
            source="stub",
            lines=[
                TranscriptLine(id=f"ln_{i:04d}", start_ms=a, end_ms=b, text=t)
                for i, (a, b, t) in enumerate(self.lines, 1)
            ],
        )
        write_model(ctx.out_dir / TRANSCRIPT_FILE, doc)
        return StageOutput()


def letters(text: str) -> list[tuple[str, str]]:
    """(first, last) place letter of every segment listed in a boundaries prompt."""
    blocks = re.split(r"\nS\d+（", "\n" + text.split("片段：", 1)[1].split("请为每个衔接处", 1)[0])[
        1:
    ]
    out = []
    for blk in blocks:
        first = re.search(r"开头：(\w)·", blk)
        last = re.search(r"结尾：(\w)·", blk)
        assert first is not None
        out.append((first.group(1), (last or first).group(1)))
    return out


class Rig:
    def __init__(
        self,
        tmp_path: Path,
        n: int,
        shot_ms: int,
        hist_of: Any,
        place_of: Any,
        *,
        lines: Any = None,
        drop: str | None = None,
        **stage_kw: Any,
    ) -> None:
        self.requests: list[str] = []
        self.boundary_calls = 0
        self.fail_on: int | None = None
        self.skip_scene_at: str | None = None  # never summarize the scene starting at this time
        self.skip_first_scene_once = False
        self.summary_requests = 0
        self.decide = lambda prev_last, next_first: prev_last != next_first
        self.llm = FakeLLM({"scene_segment": self.reply})
        self.store = ArtifactStore(tmp_path / "a")
        self.stages = [
            StubKeyframes(n, shot_ms, hist_of),
            StubCaptions(n, place_of, drop),
            StubTranscript(lines or [(1_000, 3_000, "hello")]),
        ]
        self.stage_kw = stage_kw

    def reply(self, _task: str, messages: list[Message], _schema: Any) -> dict[str, Any]:
        text = messages[-1].content
        self.requests.append(text)
        if self.fail_on is not None and len(self.requests) == self.fail_on:
            raise LLMQuotaExhausted("window used up", provider="fake")
        if "衔接处" in text:
            self.boundary_calls += 1
            segs = letters(text)
            return {
                "joins": [
                    {"after": i + 1, "new_scene": self.decide(segs[i][1], segs[i + 1][0])}
                    for i in range(len(segs) - 1)
                ]
            }
        self.summary_requests += 1
        starts = re.findall(r"^场景 \d+（(\d\d:\d\d)–", text, flags=re.M)
        out = []
        for i, start in enumerate(starts):
            if start == self.skip_scene_at or (
                self.skip_first_scene_once and self.summary_requests == 1 and i == 1
            ):
                continue
            out.append(
                {
                    "scene": i + 1,
                    "summary": f"场景 {start} 的剧情",
                    "location": " 雪山 ",
                    "importance": 1.7 if i == 0 else 0.4,
                }
            )
        return {"summaries": out}

    def engine(self, **kw: Any) -> Engine:
        stage = ScenesStage(self.llm, {"scene_segment": "fake/m"}, **self.stage_kw)
        return Engine(self.store, [*self.stages, stage], **kw)

    def run(self, **kw: Any):  # type: ignore[no-untyped-def]
        return self.engine(**kw).ensure("analysis.scenes", SCOPE)


def three_places(i: int) -> str:
    return "ABC"[0 if i < 12 else 1 if i < 26 else 2]


def read(art: Any) -> Scenes:
    return art.read_model(SCENES_FILE, Scenes)


def rig3(tmp_path: Path, hist_of: Any = None, **kw: Any) -> Rig:
    # 40 shots of 10 s; the colours change only where the place changes (12 | 26)
    hist_of = hist_of or (lambda i: hist(0 if i < 12 else 21 if i < 26 else 42))
    return Rig(tmp_path, 40, 10_000, hist_of, three_places, **kw)


def test_scenes_follow_the_places_and_partition_the_shots(tmp_path: Path) -> None:
    r = rig3(tmp_path)
    art = r.run()
    scenes = read(art).scenes
    assert [s.id for s in scenes] == ["sc_001", "sc_002", "sc_003"]
    assert [(s.shot_ids[0], s.shot_ids[-1]) for s in scenes] == [
        ("sh_0000", "sh_0011"),
        ("sh_0012", "sh_0025"),
        ("sh_0026", "sh_0039"),
    ]
    assert [s.shot_ids for s in scenes] == [
        [f"sh_{i:04d}" for i in range(a, b)] for a, b in [(0, 12), (12, 26), (26, 40)]
    ]
    assert (scenes[0].start_ms, scenes[0].end_ms, scenes[2].end_ms) == (0, 120_000, 400_000)
    assert all(a.end_ms == b.start_ms for a, b in pairwise(scenes))
    assert art.meta == {"shots": 40, "candidates": 2, "scenes": 3}


def test_summary_fields_are_cleaned_and_dialogue_is_assigned_to_scenes(tmp_path: Path) -> None:
    r = rig3(
        tmp_path,
        lines=[(1_000, 3_000, "in one"), (130_000, 131_000, "in two"), (399_000, 399_500, "last")],
    )
    scenes = read(r.run()).scenes
    assert [s.summary for s in scenes] == [
        "场景 00:00 的剧情",
        "场景 02:00 的剧情",
        "场景 04:20 的剧情",
    ]
    assert scenes[0].location == "雪山" and scenes[0].importance == 1.0  # 1.7 clamped
    assert scenes[1].importance == 0.4
    assert [s.line_ids for s in scenes] == [["ln_0001"], ["ln_0002"], ["ln_0003"]]
    assert all(s.characters == [] for s in scenes)  # no faces yet


def test_the_model_merges_candidates_inside_a_scene(tmp_path: Path) -> None:
    # the colours also change at 5 | 6, inside place A: a candidate the model must reject
    r = rig3(
        tmp_path, hist_of=lambda i: hist(0 if i < 6 else 7 if i < 12 else 21 if i < 26 else 42)
    )
    art = r.run()
    assert art.meta["candidates"] == 3 and art.meta["scenes"] == 3
    assert [len(s.shot_ids) for s in read(art).scenes] == [12, 14, 14]


def test_the_prompt_shows_segments_with_descriptions_and_dialogue(tmp_path: Path) -> None:
    r = rig3(tmp_path, lines=[(1_000, 3_000, "Where are you going?")])
    r.run()
    boundaries = next(t for t in r.requests if "衔接处" in t)
    assert (
        "S1（00:00–02:00，12 个镜头）" in boundaries
        and "S3（04:20–06:40，14 个镜头）" in boundaries
    )
    assert "开头：A·画面 0" in boundaries and "结尾：A·画面 11" in boundaries
    assert "台词：Where are you going?" in boundaries
    assert "台词：（无台词）" in boundaries
    summary = next(t for t in r.requests if "场景 1（" in t)
    assert "- [00:00] A·画面 0" in summary


def test_undecided_joins_stay_cuts(tmp_path: Path) -> None:
    r = rig3(
        tmp_path, hist_of=lambda i: hist(0 if i < 6 else 7 if i < 12 else 21 if i < 26 else 42)
    )
    r.decide = lambda prev_last, next_first: True
    art = r.run()  # every candidate is kept
    assert art.meta["scenes"] == 4

    r2 = rig3(
        tmp_path / "x",
        hist_of=lambda i: hist(0 if i < 6 else 7 if i < 12 else 21 if i < 26 else 42),
    )
    original = r2.reply

    def silent(task: str, messages: list[Message], schema: Any) -> dict[str, Any]:
        out = original(task, messages, schema)
        return (
            {"joins": []} if "joins" in out else out
        )  # the model answers nothing about boundaries

    r2.llm = FakeLLM({"scene_segment": silent})
    assert r2.run().meta["scenes"] == 4  # the candidates stand


def test_many_candidates_are_decided_window_by_window(tmp_path: Path) -> None:
    # 30 shots of 25 s, every boundary a strong visual change -> 30 segments; places change every 5
    r = Rig(
        tmp_path,
        30,
        25_000,
        lambda i: hist(i),
        lambda i: "ABCDEF"[i // 5],
        window_size=8,
        window_overlap=2,
    )
    art = r.run()
    scenes = read(art).scenes
    assert art.meta["candidates"] == 29
    assert [len(s.shot_ids) for s in scenes] == [5] * 6  # decided across several windows
    assert r.boundary_calls == 5  # windows of 8 segments, 2 overlapping


def test_a_failed_run_resumes_with_only_the_missing_requests(tmp_path: Path) -> None:
    r = Rig(tmp_path, 30, 25_000, lambda i: hist(i), lambda i: "ABCDEF"[i // 5])
    r.fail_on = 3  # quota runs out on the third request
    with pytest.raises(LLMQuotaExhausted):
        r.run()
    assert len(r.requests) == 3
    r.fail_on = None
    r.requests.clear()
    r.boundary_calls = 0
    art = r.run()
    assert [len(s.shot_ids) for s in read(art).scenes] == [5] * 6
    assert r.boundary_calls == 3  # windows 3-5 only: the first two were kept
    work = r.store.root / ".work" / "analysis.scenes"
    assert not work.exists() or not any(work.iterdir())


def test_a_summary_the_model_skips_is_asked_for_again(tmp_path: Path) -> None:
    r = rig3(tmp_path)
    r.skip_first_scene_once = True  # the first answer leaves out the second scene
    scenes = read(r.run()).scenes
    assert len(scenes) == 3 and all(s.summary for s in scenes)
    assert r.summary_requests == 2  # the batch, then the one scene that was missing


def test_a_scene_never_summarized_fails_the_stage(tmp_path: Path) -> None:
    r = rig3(tmp_path)
    r.skip_scene_at = "02:00"
    with pytest.raises(ScenesError, match="did not summarize"):
        r.run()


def test_slivers_left_by_the_model_are_folded_away(tmp_path: Path) -> None:
    # 14 shots of 2 s; the last shot changes place, so the model says "new scene" - but that
    # scene would last 2 s, far below the minimum: it is folded into the previous one
    r = Rig(
        tmp_path, 14, 2_000, lambda i: hist(0 if i < 13 else 9), lambda i: "A" if i < 13 else "B"
    )
    art = r.run()
    assert art.meta["candidates"] == 1
    (only,) = read(art).scenes
    assert len(only.shot_ids) == 14 and (only.start_ms, only.end_ms) == (0, 28_000)


def test_missing_descriptions_are_an_error(tmp_path: Path) -> None:
    r = rig3(tmp_path, drop="sh_0007")
    with pytest.raises(ScenesError, match="sh_0007"):
        r.run()


def test_a_second_run_is_a_cache_hit(tmp_path: Path) -> None:
    r = rig3(tmp_path)
    first = r.run()
    calls = len(r.requests)
    assert r.run().cache_key == first.cache_key and len(r.requests) == calls
