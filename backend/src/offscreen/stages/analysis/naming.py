"""naming: characters + dialogue + pictures -> names the film itself gives them (M3-09).

One request (task `character_name`, prompt `character_name`) shows the model, for every
character, a few face crops, a handful of representative shots (time, description, nearby
dialogue) and the film's dialogue, and asks who each one is. A name is only accepted with an
`evidence` quote that really is a line of the film: no guessing from cast lists or outside
knowledge. Names that fail the check get one repair round; what still fails stays unnamed (a
person can name the character later; the stage does not invent). `name_source` is "ai" - human
names live in the revision layer and win when read.

The characters artifact is copied through (thumbnails and centres included), so everything
downstream reads one self-contained document."""

from __future__ import annotations

import base64
import shutil
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from offscreen.algo.characters import check_names
from offscreen.algo.shot_captions import dialogue_near
from offscreen.algo.story import fmt_clock
from offscreen.domain.index import Captions, Cast, Characters, Shot, Shots, Transcript
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.prompts import render, template_version
from offscreen.providers.ports import LLM, Message
from offscreen.stages.analysis.captions import CAPTIONS_FILE
from offscreen.stages.analysis.characters import (
    CAST_FILE,
    CENTROIDS_FILE,
    CHARACTERS_FILE,
)
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.store.files import write_model

NAME_TASK = "character_name"
MAX_CHARACTERS = 12
"""The most frequent characters are named; the rest stay as they are."""
SAMPLES = 6
"""Representative shots per character."""
MAX_DIALOGUE_LINES = 600
MAX_SAMPLE_DIALOGUE = 160
ATTEMPTS = 2


class NameItem(BaseModel):
    id: str
    name: str | None = None
    aliases: list[str] = []
    role: str | None = None
    bio: str | None = None
    evidence: str | None = None


class NameReply(BaseModel):
    characters: list[NameItem] = Field(min_length=0)


def _data_url(path: Any) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _blank(value: str | None) -> str | None:
    return value.strip() or None if value is not None else None


class NamingStage(Stage):
    name = "analysis.naming"
    version = 1
    lane: Lane = "api"

    def __init__(self, llm: LLM, models: Mapping[str, str] | None = None) -> None:
        self.llm = llm
        self.models = dict(models or {})
        """`task -> "provider/model"`; part of the cache key."""

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [
            ArtifactRef("analysis.characters", scope),
            ArtifactRef("analysis.keyframes", scope),
            ArtifactRef("analysis.transcript", scope),
            ArtifactRef("analysis.captions", scope),
        ]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {
            "asset_id": scope["asset_id"],
            "max_characters": MAX_CHARACTERS,
            "samples": SAMPLES,
        }

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "models": {NAME_TASK: self.models.get(NAME_TASK)},
            "prompts": {"character_name": template_version("character_name")},
        }

    def run(self, ctx: StageContext) -> StageOutput:
        chars_art = ctx.input("analysis.characters")
        doc = chars_art.read_model(CHARACTERS_FILE, Characters)
        (ctx.out_dir / "faces").mkdir()
        for c in doc.characters:  # thumbnails and centres travel with the document
            for rel in c.face_cluster.thumbnails if c.face_cluster else []:
                shutil.copyfile(chars_art.path(rel), ctx.out_dir / rel)
        shutil.copyfile(chars_art.path(CENTROIDS_FILE), ctx.out_dir / CENTROIDS_FILE)

        named: dict[str, NameItem] = {}
        if doc.characters:
            named = self._ask_all(ctx, doc)
        if ctx.is_canceled():
            raise StageCanceled(self.name)

        out: list[Any] = []
        for c in doc.characters:
            item = named.get(c.id)
            name = _blank(item.name) if item else None
            out.append(
                c.model_copy(
                    update={
                        "name": name,
                        "name_source": "ai" if name else None,
                        "aliases": [a for a in (item.aliases if item else []) if a.strip()],
                        "role": _blank(item.role) if item else None,
                        "bio": _blank(item.bio) if item else None,
                    }
                )
            )
        write_model(
            ctx.out_dir / CHARACTERS_FILE, Characters(asset_id=doc.asset_id, characters=out)
        )
        ctx.progress(1.0, "done")
        return StageOutput(
            meta={
                "characters": len(out),
                "named": sum(1 for c in out if c.name),
                "asked": len(named),
            }
        )

    # ---- the request ---------------------------------------------------------------------
    def _ask_all(self, ctx: StageContext, doc: Characters) -> dict[str, NameItem]:
        chars_art = ctx.input("analysis.characters")
        frames = ctx.input("analysis.keyframes")
        shots = {s.id: s for s in frames.read_model(SHOTS_FILE, Shots).shots}
        lines = ctx.input("analysis.transcript").read_model(TRANSCRIPT_FILE, Transcript).lines
        captions = {
            c.shot_id: c.caption
            for c in ctx.input("analysis.captions").read_model(CAPTIONS_FILE, Captions).captions
        }
        cast = chars_art.read_model(CAST_FILE, Cast)

        asked = doc.characters[:MAX_CHARACTERS]
        images: list[str] = []
        entries: list[dict[str, Any]] = []
        for c in asked:
            thumbs = c.face_cluster.thumbnails if c.face_cluster else []
            first = len(images) + 1
            images += [_data_url(chars_art.path(t)) for t in thumbs]
            appearances = [
                (m.share, shots[s.shot_id])
                for s in cast.shots
                for m in s.characters
                if m.character_id == c.id and s.shot_id in shots
            ]
            appearances.sort(key=lambda a: (-a[0], a[1].start_ms))
            chosen = _spread(appearances, SAMPLES)
            entries.append(
                {
                    "id": c.id,
                    "shots": len(appearances),
                    "first_image": first,
                    "last_image": max(first, len(images)),
                    "samples": [
                        {
                            "start": fmt_clock(shot.start_ms),
                            "caption": captions.get(shot.id, ""),
                            "dialogue": dialogue_near(
                                lines, shot.start_ms, shot.end_ms, max_chars=MAX_SAMPLE_DIALOGUE
                            ),
                        }
                        for shot in chosen
                    ],
                }
            )
        shown = lines[:MAX_DIALOGUE_LINES]
        prompt = render(
            "character_name",
            characters=entries,
            lines=[{"start": fmt_clock(x.start_ms), "text": x.text.strip()} for x in shown],
        )
        ids = [c.id for c in asked]
        dialogue = [x.text for x in lines]

        messages = [Message("user", prompt.text, images=tuple(images))]
        ctx.progress(0.1, f"naming {len(ids)} characters")
        best: dict[str, NameItem] = {}
        for attempt in range(ATTEMPTS):
            if ctx.is_canceled():
                raise StageCanceled(self.name)
            reply = self.llm.generate(
                NAME_TASK,
                messages,
                NameReply,
                prompt_version=prompt.version,
                max_tokens=600 + 400 * len(ids),
            )
            items = {i.id: i for i in reply.characters if i.id in ids}
            for (
                cid,
                item,
            ) in items.items():  # keep the earlier answer for a character the model dropped
                best[cid] = item
            problems = check_names(
                {cid: (i.name, i.evidence) for cid, i in items.items()}, ids, dialogue
            )
            if not problems:
                return best
            if attempt + 1 < ATTEMPTS:
                messages = [
                    *messages,
                    Message("assistant", reply.model_dump_json()),
                    Message(
                        "user",
                        "上一次输出有以下问题：\n- "
                        + "\n- ".join(f"{cid}：{p}" for cid, ps in problems.items() for p in ps)
                        + "\n请重新输出所有人物的完整 JSON；没有台词依据的 name 请设为 null。",
                    ),
                ]
        # still wrong: a name without a real quote is not kept
        for cid in problems:
            if cid in best:
                best[cid] = best[cid].model_copy(update={"name": None, "evidence": None})
        return best


def _spread(ranked: list[tuple[float, Shot]], count: int) -> list[Shot]:
    """The `count` best-ranked shots (`ranked` is best first), in time order."""
    return sorted((shot for _, shot in ranked[:count]), key=lambda s: s.start_ms)
