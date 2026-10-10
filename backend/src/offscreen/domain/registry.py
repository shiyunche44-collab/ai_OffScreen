"""Every persisted document type, by name. Used for JSON Schema export and tests."""

from __future__ import annotations

from pydantic import BaseModel

from offscreen.domain.artifact import Manifest
from offscreen.domain.asset import MediaAsset
from offscreen.domain.index import (
    Captions,
    Cast,
    CharacterOverrides,
    Characters,
    CutAnnotations,
    Faces,
    Scenes,
    SelectionAnnotations,
    ShotIndex,
    Shots,
    SpriteSheets,
    Story,
    Transcript,
    VisualSignatures,
)
from offscreen.domain.job import Job
from offscreen.domain.plan import EditPlan
from offscreen.domain.project import Project
from offscreen.domain.script import Script, ScriptOutline, ScriptReview
from offscreen.domain.timeline import Timeline
from offscreen.domain.voice import VoiceLibrary

DOCUMENTS: dict[str, type[BaseModel]] = {
    "asset": MediaAsset,
    "transcript": Transcript,
    "shots": Shots,
    "signatures": VisualSignatures,
    "sprites": SpriteSheets,
    "captions": Captions,
    "faces": Faces,
    "cut_annotations": CutAnnotations,
    "selection_annotations": SelectionAnnotations,
    "shot_index": ShotIndex,
    "cast": Cast,
    "scenes": Scenes,
    "characters": Characters,
    "characters_overrides": CharacterOverrides,
    "story": Story,
    "outline": ScriptOutline,
    "script": Script,
    "script_review": ScriptReview,
    "plan": EditPlan,
    "timeline": Timeline,
    "job": Job,
    "project": Project,
    "manifest": Manifest,
    "voices": VoiceLibrary,
}
