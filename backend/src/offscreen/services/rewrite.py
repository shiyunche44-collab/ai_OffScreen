"""Rewriting one segment of a project's script: the job that does it.

The script version the request was made against (`base_version`) is part of the job. The new
text is stored as the next version (author ai) only if that version is still the current one,
so a rewrite never buries an edit made meanwhile. Annotations on the old text of the rewritten
segment are dropped (they judged words that no longer exist); the others stay."""

from __future__ import annotations

from typing import Any, Protocol

from offscreen import styles
from offscreen.config import AppConfig
from offscreen.domain.index import Scenes, Story, Transcript
from offscreen.domain.job import Job, JobCanceled
from offscreen.domain.script import Script
from offscreen.services.errors import Conflict, InvalidInput, NotFound
from offscreen.services.pipeline import Pipeline, Providers, RunOptions
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.story import STORY_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.stages.creation.rewrite import RewriteError, rewrite_segment
from offscreen.store.db import Database
from offscreen.store.documents import DocumentStore, StaleBase


class JobRun(Protocol):
    def progress(self, frac: float, msg: str = "") -> None: ...

    def is_canceled(self) -> bool: ...


def run_rewrite(
    cfg: AppConfig,
    db: Database,
    docs: DocumentStore,
    providers: Providers,
    job: Job,
    opts: RunOptions,
    ctx: JobRun,
) -> None:
    scope: dict[str, Any] = job.scope
    asset_id, project_id = str(scope["asset_id"]), str(scope["project_id"])
    segment_id, instruction = str(scope["segment_id"]), str(scope["instruction"])
    base_version = int(scope["base_version"])

    script = docs.read(project_id, "script", Script, base_version)
    if script is None:
        raise NotFound(f"the script has no version {base_version}")
    try:
        preset = styles.get(script.params.style)
    except styles.StyleError as e:
        raise InvalidInput(str(e)) from e

    ctx.progress(0.05, "reading the movie's story")
    with Pipeline(cfg, providers, db=db) as pipeline:
        story = pipeline.peek("analysis.story", asset_id, opts)
        scenes = pipeline.peek("analysis.scenes", asset_id, opts)
        transcript = pipeline.peek("analysis.transcript", asset_id, opts)
    if story is None or scenes is None or transcript is None:
        raise NotFound("the movie has not been analyzed")
    if ctx.is_canceled():
        raise JobCanceled("canceled before asking the model")

    ctx.progress(0.2, "rewriting")
    try:
        new = rewrite_segment(
            providers.llm,
            script,
            segment_id,
            instruction,
            preset=preset,
            story=story.read_model(STORY_FILE, Story),
            scenes=scenes.read_model(SCENES_FILE, Scenes).scenes,
            lines=transcript.read_model(TRANSCRIPT_FILE, Transcript).lines,
        )
    except RewriteError as e:
        raise InvalidInput(str(e)) from e

    def build(doc_id: str, version: int, parent: int | None) -> Script:
        return script.model_copy(
            update={
                "id": doc_id,
                "version": version,
                "parent_version": parent,
                "author": "ai",
                "segments": [
                    s.model_copy(update={"text": new.text, "scene_refs": new.scene_refs})
                    if s.id == segment_id
                    else s
                    for s in script.segments
                ],
                "annotations": [a for a in script.annotations if a.segment_id != segment_id],
            }
        )

    try:
        docs.append(project_id, "script", base_version, build, author="ai")
    except StaleBase as e:
        raise Conflict(
            f"the script changed while {segment_id} was being rewritten (it was version "
            f"{base_version}, now {e.current}); ask again to build on the current one"
        ) from e
    ctx.progress(1.0, "done")
