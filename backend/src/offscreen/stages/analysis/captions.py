"""captions: keyframes + dialogue -> captions.json, a description of every shot (M3-06).

Shots go to the vision model (task `shot_caption`) in consecutive batches, each with its three
keyframes and the dialogue around it. The reply is checked shot by shot: shots the model skipped
are asked for again once, then the stage fails rather than invent descriptions.

The work is the expensive part of analysis (and subscription plans have quota windows), so every
finished batch is written to the stage's work directory at once. A run that dies - quota used up,
rate limits, a killed worker - resumes there: the next attempt only pays for what is missing. The
estimated token cost is reported before the first call."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any, get_args

from pydantic import BaseModel, Field, field_validator

from offscreen.algo.shot_captions import (
    batches,
    dialogue_near,
    estimate_tokens,
    missing,
)
from offscreen.algo.story import fmt_clock
from offscreen.domain.index import Captions, Shot, ShotCaption, Shots, ShotSize, Transcript
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.prompts import render, template_version
from offscreen.providers.ports import LLM, Message
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.store.files import read_model, write_model

CAPTIONS_FILE = "captions.json"
CAPTION_TASK = "shot_caption"
DEFAULT_BATCH = 8
FRAMES_PER_SHOT = 3
SHOT_SIZES = frozenset(get_args(ShotSize))
MAX_OUT_TOKENS_PER_SHOT = 220


class CaptionsError(RuntimeError):
    pass


class CaptionItem(BaseModel):
    """One shot as the model returns it. Lenient where models drift (an unknown shot size becomes
    "other", blank strings become empty); strict about what matters: the id and the caption."""

    shot_id: str
    caption: str = Field(min_length=1)
    shot_size: str = "other"
    action: str | None = None
    emotion: str | None = None
    has_onscreen_text: bool = False
    is_credits: bool = False

    @field_validator("caption")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("empty caption")
        return v

    @field_validator("shot_size")
    @classmethod
    def _known_size(cls, v: str) -> str:
        v = v.strip().lower()
        return v if v in SHOT_SIZES else "other"

    @field_validator("action", "emotion")
    @classmethod
    def _blank_is_none(cls, v: str | None) -> str | None:
        return v.strip() or None if v else None


class CaptionReply(BaseModel):
    captions: list[CaptionItem]


class CaptionsStage(Stage):
    name = "analysis.captions"
    version = 1
    lane: Lane = "api"

    def __init__(
        self,
        llm: LLM,
        models: Mapping[str, str] | None = None,
        shots_per_request: int = DEFAULT_BATCH,
    ) -> None:
        self.llm = llm
        self.models = dict(models or {})
        self.batch_size = shots_per_request

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.keyframes", scope), ArtifactRef("analysis.transcript", scope)]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {
            "asset_id": scope["asset_id"],
            "shots_per_request": self.batch_size,
            "frames_per_shot": FRAMES_PER_SHOT,
            "language": "zh",
        }

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "models": {CAPTION_TASK: self.models.get(CAPTION_TASK)},
            "prompts": {CAPTION_TASK: template_version("shot_caption")},
        }

    def run(self, ctx: StageContext) -> StageOutput:
        keyframes = ctx.input("analysis.keyframes")
        shots = keyframes.read_model(SHOTS_FILE, Shots)
        lines = sorted(
            ctx.input("analysis.transcript").read_model(TRANSCRIPT_FILE, Transcript).lines,
            key=lambda x: (x.start_ms, x.end_ms),
        )
        if not shots.shots:
            raise CaptionsError("no shots to describe")

        dialogue = {s.id: dialogue_near(lines, s.start_ms, s.end_ms) for s in shots.shots}
        estimate = estimate_tokens(
            len(shots.shots),
            batch_size=self.batch_size,
            frames_per_shot=FRAMES_PER_SHOT,
            dialogue_chars=sum(len(d) for d in dialogue.values()),
        )
        plan = batches(shots.shots, self.batch_size)
        ctx.progress(
            0.0, f"{len(shots.shots)} shots in {len(plan)} requests, about {estimate} tokens"
        )

        done: dict[str, ShotCaption] = {}
        resumed = 0
        for i, batch in enumerate(plan):
            if ctx.is_canceled():
                raise StageCanceled(self.name)
            saved = ctx.work_dir / f"batch_{i:05d}.json"
            captions = self._load_batch(saved, shots.asset_id, batch)
            if captions is None:
                captions = self._describe(batch, dialogue, lambda rel: keyframes.path(rel))
                write_model(saved, Captions(asset_id=shots.asset_id, captions=captions))
            else:
                resumed += 1
            done.update({c.shot_id: c for c in captions})
            ctx.progress(
                0.98 * (i + 1) / len(plan), f"described {len(done)}/{len(shots.shots)} shots"
            )

        write_model(
            ctx.out_dir / CAPTIONS_FILE,
            Captions(asset_id=shots.asset_id, captions=[done[s.id] for s in shots.shots]),
        )
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={
                "shots": len(shots.shots),
                "requests": len(plan),
                "resumed_requests": resumed,
                "estimated_tokens": estimate,
            }
        )

    # ---- one batch ---------------------------------------------------------------------
    @staticmethod
    def _load_batch(path: Any, asset_id: str, batch: list[Shot]) -> list[ShotCaption] | None:
        """A batch finished by an earlier attempt, if it is intact and is exactly this batch."""
        if not path.is_file():
            return None
        try:
            saved = read_model(path, Captions)
        except ValueError:
            return None
        if saved.asset_id != asset_id or [c.shot_id for c in saved.captions] != [
            s.id for s in batch
        ]:
            return None
        return saved.captions

    def _describe(
        self, batch: list[Shot], dialogue: Mapping[str, str], frame_path: Any
    ) -> list[ShotCaption]:
        got = self._ask(batch, dialogue, frame_path)
        again = [s for s in batch if s.id in missing([s.id for s in batch], list(got))]
        if again:  # the model skipped some shots: ask for exactly those once more
            got.update(self._ask(again, dialogue, frame_path))
        gone = missing([s.id for s in batch], list(got))
        if gone:
            raise CaptionsError(f"the model did not describe shots {', '.join(gone)}")
        return [got[s.id] for s in batch]

    def _ask(
        self, batch: list[Shot], dialogue: Mapping[str, str], frame_path: Any
    ) -> dict[str, ShotCaption]:
        images: list[str] = []
        entries: list[dict[str, Any]] = []
        for shot in batch:
            first = len(images) + 1
            images += [_data_url(frame_path(rel)) for rel in shot.keyframes]
            entries.append(
                {
                    "id": shot.id,
                    "start": fmt_clock(shot.start_ms),
                    "end": fmt_clock(shot.end_ms),
                    "first_image": first,
                    "last_image": len(images),
                    "dialogue": dialogue.get(shot.id, ""),
                }
            )
        prompt = render("shot_caption", shots=entries)
        reply = self.llm.generate(
            CAPTION_TASK,
            [Message("user", prompt.text, images=tuple(images))],
            CaptionReply,
            prompt_version=prompt.version,
            max_tokens=512 + MAX_OUT_TOKENS_PER_SHOT * len(batch),
        )
        wanted = {s.id for s in batch}
        out: dict[str, ShotCaption] = {}
        for item in reply.captions:
            if item.shot_id in wanted and item.shot_id not in out:  # ignore strays and repeats
                out[item.shot_id] = ShotCaption(
                    shot_id=item.shot_id,
                    caption=item.caption,
                    shot_size=item.shot_size,
                    action=item.action,
                    emotion=item.emotion,
                    has_onscreen_text=item.has_onscreen_text,
                    is_credits=item.is_credits,
                )
        return out


def _data_url(path: Any) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
