"""The captions stage: batching, prompts with images, checking replies, and resuming."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

import pytest

from offscreen.domain.index import Captions, Shot, Shots, Transcript, TranscriptLine
from offscreen.engine import (
    ArtifactStore,
    Engine,
    Scope,
    Stage,
    StageCanceled,
    StageContext,
    StageOutput,
)
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.providers.ports import LLMQuotaExhausted, Message
from offscreen.stages.analysis.captions import (
    CAPTIONS_FILE,
    CaptionsError,
    CaptionsStage,
)
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.store.files import write_model

ASSET = "ast_t1"
SCOPE = {"asset_id": ASSET}


class StubKeyframes(Stage):
    """Shots of 10 s each with three fake frames per shot (the bytes name their shot)."""

    name = "analysis.keyframes"
    version = 1
    lane = "cpu"

    def __init__(self, n: int) -> None:
        self.n = n

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"n": self.n}

    def run(self, ctx: StageContext) -> StageOutput:
        (ctx.out_dir / "kf").mkdir()
        shots = []
        for i in range(self.n):
            sid = f"sh_{i:04d}"
            rels = []
            for c in "abc":
                rel = f"kf/{sid}_{c}.jpg"
                (ctx.out_dir / rel).write_bytes(f"jpeg:{sid}:{c}".encode())
                rels.append(rel)
            shots.append(Shot(id=sid, start_ms=i * 10_000, end_ms=(i + 1) * 10_000, keyframes=rels))
        write_model(ctx.out_dir / SHOTS_FILE, Shots(asset_id=ASSET, shots=shots))
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


def shot_ids(messages: list[Message]) -> list[str]:
    return re.findall(r"镜头 (sh_\d+)（", messages[-1].content)


def answer(ids: list[str], **overrides: Any) -> dict[str, Any]:
    return {
        "captions": [
            {
                "shot_id": sid,
                "caption": f"画面 {sid}",
                "shot_size": "medium",
                "action": "走",
                "emotion": "平静",
                **overrides,
            }
            for sid in ids
        ]
    }


class Setup:
    def __init__(
        self, tmp_path: Path, n: int = 20, per: int = 8, handler: Any = None, lines: Any = None
    ) -> None:
        self.requests: list[list[str]] = []
        self.messages: list[list[Message]] = []
        self.handler = handler or (lambda ids, call: answer(ids))

        def reply(_task: str, messages: list[Message], _schema: Any) -> dict[str, Any]:
            ids = shot_ids(messages)
            self.requests.append(ids)
            self.messages.append(messages)
            return self.handler(ids, len(self.requests))  # type: ignore[no-any-return]

        self.llm = FakeLLM({"shot_caption": reply})
        self.store = ArtifactStore(tmp_path / "a")
        self.n, self.per = n, per
        self.lines = lines or [(5_000, 8_000, "Where are you going?")]

    def engine(self, **kw: Any) -> Engine:
        stage = CaptionsStage(self.llm, {"shot_caption": "fake/vlm"}, kw.pop("per", self.per))
        return Engine(self.store, [StubKeyframes(self.n), StubTranscript(self.lines), stage], **kw)

    def run(self, **kw: Any):  # type: ignore[no-untyped-def]
        return self.engine(**kw).ensure("analysis.captions", SCOPE)


def read(art: Any) -> Captions:
    return art.read_model(CAPTIONS_FILE, Captions)


def test_every_shot_is_described_in_order_in_batches(tmp_path: Path) -> None:
    s = Setup(tmp_path, n=20, per=8)
    art = s.run()
    caps = read(art).captions
    assert [c.shot_id for c in caps] == [f"sh_{i:04d}" for i in range(20)]
    assert all(c.caption.startswith("画面 ") and c.shot_size == "medium" for c in caps)
    assert [len(r) for r in s.requests] == [8, 8, 4]
    assert art.meta["requests"] == 3 and art.meta["resumed_requests"] == 0
    assert art.meta["estimated_tokens"] > 20 * 3 * 400


def test_the_request_carries_three_frames_per_shot_and_the_dialogue_around_it(
    tmp_path: Path,
) -> None:
    s = Setup(
        tmp_path,
        n=3,
        per=8,
        lines=[(5_000, 8_000, "Where are you going?"), (95_000, 96_000, "far")],
    )
    s.run()
    (messages,) = s.messages
    (msg,) = [m for m in messages if m.images]
    assert len(msg.images) == 9
    decoded = [base64.b64decode(u.split(",", 1)[1]).decode() for u in msg.images]
    assert decoded[:4] == ["jpeg:sh_0000:a", "jpeg:sh_0000:b", "jpeg:sh_0000:c", "jpeg:sh_0001:a"]
    assert all(u.startswith("data:image/jpeg;base64,") for u in msg.images)
    text = msg.content
    assert "镜头 sh_0000（00:00–00:10）图 1–3：Where are you going?" in text
    assert "镜头 sh_0001（00:10–00:20）图 4–6：（无台词）" in text  # 10 s is >2 s after the line
    assert "镜头 sh_0002（00:20–00:30）图 7–9" in text


def test_fields_are_normalised_and_mapped(tmp_path: Path) -> None:
    def handler(ids: list[str], _n: int) -> dict[str, Any]:
        return {
            "captions": [
                {
                    "shot_id": ids[0],
                    "caption": "  片头标题  ",
                    "shot_size": "Close_Up",
                    "action": "  ",
                    "emotion": None,
                    "is_credits": True,
                },
                {
                    "shot_id": ids[1],
                    "caption": "路牌",
                    "shot_size": "dutch-angle",
                    "has_onscreen_text": True,
                    "extra_field": 1,
                },
            ]
        }

    s = Setup(tmp_path, n=2, handler=handler)
    a, b = read(s.run()).captions
    assert (a.caption, a.shot_size, a.action, a.emotion, a.is_credits) == (
        "片头标题",
        "close_up",
        None,
        None,
        True,
    )
    assert (b.shot_size, b.has_onscreen_text) == (
        "other",
        True,
    )  # an unknown size is "other", extras ignored


def test_skipped_shots_are_asked_for_again_once(tmp_path: Path) -> None:
    def handler(ids: list[str], call: int) -> dict[str, Any]:
        return answer(ids[:-1] if call == 1 else ids)  # the model forgets the last shot first

    s = Setup(tmp_path, n=5, per=8, handler=handler)
    caps = read(s.run()).captions
    assert [c.shot_id for c in caps] == [f"sh_{i:04d}" for i in range(5)]
    assert s.requests == [[f"sh_{i:04d}" for i in range(5)], ["sh_0004"]]


def test_a_shot_the_model_never_describes_fails_the_stage(tmp_path: Path) -> None:
    s = Setup(tmp_path, n=3, handler=lambda ids, _n: answer([i for i in ids if i != "sh_0001"]))
    with pytest.raises(CaptionsError, match="sh_0001"):
        s.run()
    assert len(s.requests) == 2  # the retry, then give up


def test_unknown_and_repeated_shot_ids_in_a_reply_are_ignored(tmp_path: Path) -> None:
    def handler(ids: list[str], _n: int) -> dict[str, Any]:
        good = answer(ids)["captions"]
        return {
            "captions": [
                *good,
                {"shot_id": "sh_9999", "caption": "stray"},
                {**good[0], "caption": "again"},
            ]
        }

    s = Setup(tmp_path, n=3, handler=handler)
    caps = read(s.run()).captions
    assert len(caps) == 3 and caps[0].caption == "画面 sh_0000"  # the first answer wins
    assert len(s.requests) == 1


def test_a_failed_run_resumes_with_only_the_missing_batches(tmp_path: Path) -> None:
    broken = {"on": True}

    def handler(ids: list[str], call: int) -> dict[str, Any]:
        if broken["on"] and call == 2:
            raise LLMQuotaExhausted("window used up", provider="fake")
        return answer(ids)

    s = Setup(tmp_path, n=20, per=8, handler=handler)
    with pytest.raises(LLMQuotaExhausted):
        s.run()
    assert len(s.requests) == 2  # batch 1 done and saved, batch 2 refused

    broken["on"] = False
    s.requests.clear()
    art = s.run()
    assert [len(r) for r in s.requests] == [8, 4]  # batch 1 was not paid for again
    assert s.requests[0][0] == "sh_0008"
    assert art.meta["resumed_requests"] == 1
    assert [c.shot_id for c in read(art).captions] == [f"sh_{i:04d}" for i in range(20)]
    work = s.store.root / ".work" / "analysis.captions"
    assert not work.exists() or not any(work.iterdir())  # cleaned up after success


def test_corrupt_or_foreign_saved_batches_are_recomputed(tmp_path: Path) -> None:
    s = Setup(tmp_path, n=4, per=2)
    engine = s.engine()
    # Build the stage's work directory by hand: one garbled batch and one that is another batch.
    stage = engine.stages["analysis.captions"]
    kf = engine.ensure("analysis.keyframes", SCOPE)
    tr = engine.ensure("analysis.transcript", SCOPE)
    from offscreen.engine import compute_cache_key

    cache_key = compute_cache_key(
        stage.name,
        stage.version,
        [kf.content_hash, tr.content_hash],
        stage.params(SCOPE),
        stage.provider_info(SCOPE),
    )
    work = s.store.work_dir(stage.name, cache_key)
    (work / "batch_00000.json").write_text("{not json")
    other = Captions(asset_id=ASSET, captions=[])
    write_model(work / "batch_00001.json", other)
    art = engine.ensure("analysis.captions", SCOPE)
    assert len(s.requests) == 2 and art.meta["resumed_requests"] == 0


def test_a_cached_result_is_reused_and_other_batch_sizes_get_their_own(tmp_path: Path) -> None:
    s = Setup(tmp_path, n=6, per=3)
    first = s.run()
    calls = len(s.requests)
    assert s.run().cache_key == first.cache_key and len(s.requests) == calls  # cache hit
    other = s.run(per=2)
    assert other.cache_key != first.cache_key and len(s.requests) > calls


def test_cancel_between_batches_keeps_finished_work(tmp_path: Path) -> None:
    s = Setup(tmp_path, n=20, per=8)
    flag = {"stop": False}

    def progress(stage: str, _f: float, msg: str) -> None:
        if stage == "analysis.captions" and msg.startswith("described"):
            flag["stop"] = True

    with pytest.raises(StageCanceled):
        s.run(progress=progress, is_canceled=lambda: flag["stop"])
    assert len(s.requests) == 1
    s.requests.clear()
    s.run()
    assert [len(r) for r in s.requests] == [8, 4]  # resumed after the first batch


def test_no_shots_is_an_error(tmp_path: Path) -> None:
    s = Setup(tmp_path, n=0)
    with pytest.raises(CaptionsError, match="no shots"):
        s.run()


def test_the_estimate_is_reported_before_the_first_request(tmp_path: Path) -> None:
    s = Setup(tmp_path, n=10, per=5)
    seen: list[str] = []
    s.run(progress=lambda st, _f, m: seen.append(m) if st == "analysis.captions" else None)
    assert re.match(r"10 shots in 2 requests, about \d+ tokens", seen[0])
