"""scenes: shots + descriptions + dialogue -> scenes.json (M3-11).

Two passes. Candidate cuts are chosen where the picture changes a lot between two shots and the
dialogue pauses (`algo.scenes`, generous on purpose). The text between candidates is a *segment*;
the LLM then looks at windows of consecutive segments, with their shot descriptions and dialogue,
and decides for every join whether a new scene starts there (task `scene_segment`, prompt
`scene_boundaries`). Finally each scene gets a summary, a location and an importance (same task,
prompt `scene_summary`). Characters stay empty until faces and naming exist (M3-08 / M3-09).

Both passes save every finished request to the work directory, so a run that dies resumes with
only the missing requests (see `captions`). Answers the model leaves out fall back safely: an
undecided join stays a cut (the candidate was strong enough to be proposed); a missing summary is
asked for once more and then fails the stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from offscreen.algo.scenes import (
    MAX_SPAN_MS,
    MIN_SCENE_MS,
    Window,
    boundary_features,
    enforce_min_length,
    merge_segments,
    pick_candidates,
    plan_windows,
    segments_from_cuts,
)
from offscreen.algo.shot_captions import batches, dialogue_near
from offscreen.algo.story import fmt_clock
from offscreen.domain.index import (
    Captions,
    Scene,
    Scenes,
    Shot,
    ShotCaption,
    Shots,
    Transcript,
    TranscriptLine,
    VisualSignatures,
)
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.prompts import render, template_version
from offscreen.providers.ports import LLM, Message
from offscreen.stages.analysis.captions import CAPTIONS_FILE
from offscreen.stages.analysis.keyframes import SIGNATURES_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.store.files import read_model, write_model

SCENES_FILE = "scenes.json"
SEGMENT_TASK = "scene_segment"
WINDOW_SIZE = 8
WINDOW_OVERLAP = 2
SUMMARY_BATCH = 4
SEGMENT_DIALOGUE_CHARS = 200
SCENE_DIALOGUE_CHARS = 700
SCENE_CAPTIONS = 12
CAPTION_CHARS = 70


class ScenesError(RuntimeError):
    pass


# ---- what the model returns (lenient: unknown fields are ignored) ----------------------
class JoinReply(BaseModel):
    after: int
    new_scene: bool


class BoundaryReply(BaseModel):
    joins: list[JoinReply]


class SummaryItem(BaseModel):
    scene: int
    summary: str = Field(min_length=1)
    location: str | None = None
    importance: float = 0.5

    @field_validator("summary")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("empty summary")
        return v

    @field_validator("location")
    @classmethod
    def _blank_is_none(cls, v: str | None) -> str | None:
        return v.strip() or None if v else None

    @field_validator("importance")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return min(1.0, max(0.0, v))


class SummaryReply(BaseModel):
    summaries: list[SummaryItem]


class SavedWindow(BaseModel):
    """A finished window request, as kept in the work directory."""

    lo: int
    hi: int
    decisions: dict[int, bool]
    """Join index (over all segments) -> new scene starts there."""


class SavedSummaries(BaseModel):
    spans: list[tuple[int, int]]
    """Start / end (ms) of the scenes in this request, to check the batch is still the same."""
    items: list[SummaryItem]


class ScenesStage(Stage):
    name = "analysis.scenes"
    version = 1
    lane: Lane = "api"

    def __init__(
        self,
        llm: LLM,
        models: Mapping[str, str] | None = None,
        *,
        window_size: int = WINDOW_SIZE,
        window_overlap: int = WINDOW_OVERLAP,
        summary_batch: int = SUMMARY_BATCH,
    ) -> None:
        self.llm = llm
        self.models = dict(models or {})
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.summary_batch = summary_batch

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [
            ArtifactRef("analysis.keyframes", scope),
            ArtifactRef("analysis.captions", scope),
            ArtifactRef("analysis.transcript", scope),
        ]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {
            "asset_id": scope["asset_id"],
            "window": [self.window_size, self.window_overlap],
            "summary_batch": self.summary_batch,
            "min_scene_ms": MIN_SCENE_MS,
            "max_span_ms": MAX_SPAN_MS,
            "language": "zh",
        }

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "models": {SEGMENT_TASK: self.models.get(SEGMENT_TASK)},
            "prompts": {
                "boundaries": template_version("scene_boundaries"),
                "summary": template_version("scene_summary"),
            },
        }

    def run(self, ctx: StageContext) -> StageOutput:
        keyframes = ctx.input("analysis.keyframes")
        shots_doc = keyframes.read_model(SHOTS_FILE, Shots)
        sigs = keyframes.read_model(SIGNATURES_FILE, VisualSignatures)
        captions = ctx.input("analysis.captions").read_model(CAPTIONS_FILE, Captions)
        transcript = ctx.input("analysis.transcript").read_model(TRANSCRIPT_FILE, Transcript)
        shots = shots_doc.shots
        if not shots:
            raise ScenesError("no shots")
        caption_of = {c.shot_id: c for c in captions.captions}
        if missing := [s.id for s in shots if s.id not in caption_of]:
            raise ScenesError(f"shots without a description: {', '.join(missing[:5])}")
        frames = {s.shot_id: s.frames for s in sigs.signatures}
        lines = sorted(transcript.lines, key=lambda x: (x.start_ms, x.end_ms))
        duration_ms = shots[-1].end_ms

        # pass 1: candidate cuts and the segments between them
        feats = boundary_features(
            [s.end_ms for s in shots],
            [frames[s.id][0] for s in shots],
            [frames[s.id][-1] for s in shots],
            lines,
            duration_ms,
        )
        cuts = pick_candidates(feats, duration_ms=duration_ms)
        segments = segments_from_cuts(len(shots), cuts)
        windows = plan_windows(len(segments), self.window_size, self.window_overlap)
        ctx.progress(0.0, f"{len(cuts)} candidate cuts, {len(windows)} windows")

        # pass 2: the model decides the joins window by window
        new_scene = [True] * max(0, len(segments) - 1)  # undecided joins stay cuts
        total_steps = len(windows) + max(1, -(-len(segments) // self.summary_batch))
        for k, window in enumerate(windows):
            if ctx.is_canceled():
                raise StageCanceled(self.name)
            decided = self._window(ctx, k, window, segments, shots, caption_of, lines)
            for join, flag in decided.items():
                new_scene[join] = flag
            ctx.progress(0.9 * (k + 1) / total_steps, f"window {k + 1}/{len(windows)}")

        spans = merge_segments(segments, new_scene)
        spans = enforce_min_length(
            spans, [s.start_ms for s in shots], [s.end_ms for s in shots], MIN_SCENE_MS
        )

        # pass 3: a summary for every scene
        summaries = self._summaries(ctx, spans, shots, caption_of, lines, len(windows), total_steps)

        scenes = []
        width = max(3, len(str(len(spans))))
        for i, ((lo, hi), item) in enumerate(zip(spans, summaries, strict=True), 1):
            start_ms, end_ms = shots[lo].start_ms, shots[hi - 1].end_ms
            scenes.append(
                Scene(
                    id=f"sc_{i:0{width}d}",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    shot_ids=[s.id for s in shots[lo:hi]],
                    line_ids=[x.id for x in lines if start_ms <= x.start_ms < end_ms],
                    summary=item.summary,
                    location=item.location,
                    importance=item.importance,
                )
            )
        write_model(ctx.out_dir / SCENES_FILE, Scenes(asset_id=shots_doc.asset_id, scenes=scenes))
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={"shots": len(shots), "candidates": len(cuts), "scenes": len(scenes)}
        )

    # ---- boundaries --------------------------------------------------------------------
    def _window(
        self,
        ctx: StageContext,
        k: int,
        w: Window,
        segments: Sequence[tuple[int, int]],
        shots: Sequence[Shot],
        caption_of: Mapping[str, ShotCaption],
        lines: Sequence[TranscriptLine],
    ) -> dict[int, bool]:
        """Decisions (join index -> new scene) this window is trusted for."""
        saved = ctx.work_dir / f"window_{k:04d}.json"
        wanted = range(w.accept_lo, w.accept_hi)
        if (done := _load(saved, SavedWindow)) is not None and (done.lo, done.hi) == (w.lo, w.hi):
            return {j: f for j, f in done.decisions.items() if j in wanted}

        seg_info = [_segment_info(segments[i], shots, caption_of, lines) for i in range(w.lo, w.hi)]
        prompt = render("scene_boundaries", segments=seg_info)
        reply = self.llm.generate(
            SEGMENT_TASK,
            [Message("user", prompt.text)],
            BoundaryReply,
            prompt_version=prompt.version,
            max_tokens=256 + 40 * len(seg_info),
        )
        decisions: dict[int, bool] = {}
        for item in reply.joins:
            join = w.lo + item.after - 1  # `after` counts segments of this window from 1
            if w.lo <= join <= w.hi - 2 and join not in decisions:
                decisions[join] = item.new_scene
        write_model(saved, SavedWindow(lo=w.lo, hi=w.hi, decisions=decisions))
        return {j: f for j, f in decisions.items() if j in wanted}

    # ---- summaries ---------------------------------------------------------------------
    def _summaries(
        self,
        ctx: StageContext,
        spans: Sequence[tuple[int, int]],
        shots: Sequence[Shot],
        caption_of: Mapping[str, ShotCaption],
        lines: Sequence[TranscriptLine],
        windows_done: int,
        total_steps: int,
    ) -> list[SummaryItem]:
        out: list[SummaryItem] = []
        parts = batches(list(spans), self.summary_batch)
        for b, part in enumerate(parts):
            if ctx.is_canceled():
                raise StageCanceled(self.name)
            times = [(shots[lo].start_ms, shots[hi - 1].end_ms) for lo, hi in part]
            saved = ctx.work_dir / f"summary_{b:04d}.json"
            if (
                (done := _load(saved, SavedSummaries)) is not None
                and [tuple(t) for t in done.spans] == times
                and len(done.items) == len(part)
            ):
                out += done.items
            else:
                items = self._summarize(part, shots, caption_of, lines)
                write_model(saved, SavedSummaries(spans=times, items=items))
                out += items
            ctx.progress(
                0.9 * (windows_done + b + 1) / total_steps + 0.05,
                f"summaries {len(out)}/{len(spans)}",
            )
        return out

    def _summarize(
        self,
        part: Sequence[tuple[int, int]],
        shots: Sequence[Shot],
        caption_of: Mapping[str, ShotCaption],
        lines: Sequence[TranscriptLine],
    ) -> list[SummaryItem]:
        got = self._ask_summaries(part, shots, caption_of, lines)
        again = [i for i in range(len(part)) if i not in got]
        if again:  # the model skipped some scenes: ask for exactly those once more
            sub = self._ask_summaries([part[i] for i in again], shots, caption_of, lines)
            got.update({again[j]: item for j, item in sub.items()})
        if gone := [i + 1 for i in range(len(part)) if i not in got]:
            raise ScenesError(f"the model did not summarize scenes {gone} of a request")
        return [got[i] for i in range(len(part))]

    def _ask_summaries(
        self,
        part: Sequence[tuple[int, int]],
        shots: Sequence[Shot],
        caption_of: Mapping[str, ShotCaption],
        lines: Sequence[TranscriptLine],
    ) -> dict[int, SummaryItem]:
        info = [_scene_info(span, shots, caption_of, lines) for span in part]
        prompt = render("scene_summary", scenes=info)
        reply = self.llm.generate(
            SEGMENT_TASK,
            [Message("user", prompt.text)],
            SummaryReply,
            prompt_version=prompt.version,
            max_tokens=400 * len(part) + 256,
        )
        out: dict[int, SummaryItem] = {}
        for item in reply.summaries:
            idx = item.scene - 1
            if 0 <= idx < len(part) and idx not in out:
                out[idx] = item
        return out


# ---- prompt material -----------------------------------------------------------------
def _trim(text: str, limit: int = CAPTION_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _credits_share(
    span: tuple[int, int], shots: Sequence[Shot], caption_of: Mapping[str, ShotCaption]
) -> bool:
    lo, hi = span
    return sum(caption_of[s.id].is_credits for s in shots[lo:hi]) * 2 > hi - lo


def _segment_info(
    span: tuple[int, int],
    shots: Sequence[Shot],
    caption_of: Mapping[str, ShotCaption],
    lines: Sequence[TranscriptLine],
) -> dict[str, Any]:
    lo, hi = span
    start_ms, end_ms = shots[lo].start_ms, shots[hi - 1].end_ms

    def text(i: int) -> str:
        return _trim(caption_of[shots[i].id].caption)

    mid = (lo + hi - 1) // 2
    return {
        "start": fmt_clock(start_ms),
        "end": fmt_clock(end_ms),
        "shots": hi - lo,
        "first": text(lo),
        "middle": text(mid) if hi - lo >= 3 else "",
        "last": text(hi - 1) if hi - lo >= 2 else "",
        "dialogue": dialogue_near(
            lines, start_ms, end_ms, pad_ms=0, max_chars=SEGMENT_DIALOGUE_CHARS
        ),
        "credits": _credits_share(span, shots, caption_of),
    }


def _scene_info(
    span: tuple[int, int],
    shots: Sequence[Shot],
    caption_of: Mapping[str, ShotCaption],
    lines: Sequence[TranscriptLine],
) -> dict[str, Any]:
    lo, hi = span
    start_ms, end_ms = shots[lo].start_ms, shots[hi - 1].end_ms
    n = hi - lo
    picks = sorted(
        {
            lo + round(i * (n - 1) / max(1, min(n, SCENE_CAPTIONS) - 1))
            for i in range(min(n, SCENE_CAPTIONS))
        }
    )
    return {
        "start": fmt_clock(start_ms),
        "end": fmt_clock(end_ms),
        "shots": n,
        "captions": [
            {"at": fmt_clock(shots[i].start_ms), "text": _trim(caption_of[shots[i].id].caption)}
            for i in picks
        ],
        "dialogue": dialogue_near(
            lines, start_ms, end_ms, pad_ms=0, max_chars=SCENE_DIALOGUE_CHARS
        ),
        "credits": _credits_share(span, shots, caption_of),
    }


# ---- work files ----------------------------------------------------------------------
def _load(path: Path, cls: type[BaseModel]) -> Any:
    """A request finished by an earlier attempt, or None if it is missing or unreadable."""
    if not path.is_file():
        return None
    try:
        return read_model(path, cls)
    except ValueError:
        return None
