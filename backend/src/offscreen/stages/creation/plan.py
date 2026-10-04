"""plan: script + scenes + shots -> plan.json, with the voice-over audio (v0: naive picking).

For every narration segment the TTS port speaks the text, and the clips are cut from the
shots of the scenes it cites (`scene_refs`): shots no earlier segment used come first, in
source order, and fill the audio's duration (`algo.matching`). No scoring, no retrieval, no
speed changes: that is M5. Audio files are written under `tts/` in this artifact and
`AudioRef.file` is relative to the artifact directory, like `Shot.keyframes`. M1 has no
document store, so the plan is version 1."""

from __future__ import annotations

import hashlib
from typing import Any

from offscreen.algo.matching import MAX_CLIP_MS, MIN_CLIP_MS, Span, fill_clips, order_candidates
from offscreen.algo.tts import tts_cache_key
from offscreen.domain.index import Scenes, Shots
from offscreen.domain.job import Lane
from offscreen.domain.plan import (
    AudioRef,
    Clip,
    EditPlan,
    PlanSegment,
    ScriptRef,
    SourceAudio,
    VoiceSpec,
)
from offscreen.domain.script import Script
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.providers.ports import TTS
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.story import SCENES_FILE
from offscreen.stages.creation.script import SCRIPT_FILE
from offscreen.store.files import write_model

PLAN_FILE = "plan.json"
AUDIO_DIR = "tts"
NARRATION_SOURCE_GAIN_DB = -20.0
"""Film sound under the narration (ARCHITECTURE M1-13: original audio at -20 dB)."""


class PlanError(RuntimeError):
    pass


class PlanStage(Stage):
    name = "creation.plan"
    version = 1
    lane: Lane = "api"

    def __init__(self, tts: TTS, voice_speed: float = 1.0) -> None:
        if not 0.5 <= voice_speed <= 2.0:
            raise ValueError("voice_speed must be within 0.5-2.0")
        self.tts = tts
        self.voice_speed = voice_speed

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [
            ArtifactRef("creation.script", scope),
            ArtifactRef("analysis.story", scope),
            ArtifactRef("analysis.shots", scope),
        ]

    def params(self, scope: Scope) -> dict[str, Any]:
        return {
            "asset_id": scope["asset_id"],
            "voice_speed": self.voice_speed,
            "max_clip_ms": MAX_CLIP_MS,
            "min_clip_ms": MIN_CLIP_MS,
            "source_gain_db": NARRATION_SOURCE_GAIN_DB,
        }

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {"tts": self.tts.id}

    def run(self, ctx: StageContext) -> StageOutput:
        asset_id = ctx.scope["asset_id"]
        script = ctx.input("creation.script").read_model(SCRIPT_FILE, Script)
        scenes = {
            s.id: s for s in ctx.input("analysis.story").read_model(SCENES_FILE, Scenes).scenes
        }
        shots = {s.id: s for s in ctx.input("analysis.shots").read_model(SHOTS_FILE, Shots).shots}
        voice = VoiceSpec(voice_id=script.params.voice_id, speed=self.voice_speed)
        (ctx.out_dir / AUDIO_DIR).mkdir()

        used: set[str] = set()
        segments: list[PlanSegment] = []
        n = len(script.segments)
        for i, seg in enumerate(script.segments):
            if ctx.is_canceled():
                raise StageCanceled(self.name)
            if seg.kind != "narration":
                raise PlanError(f"{seg.id}: original-sound segments are not supported in v0")

            audio = self.tts.synthesize(seg.text, voice_id=voice.voice_id, speed=voice.speed)
            key = tts_cache_key(seg.text, voice.voice_id, voice.speed, self.tts.id)
            rel = f"{AUDIO_DIR}/{key.removeprefix('sha256:')[:16]}.{audio.format}"
            (ctx.out_dir / rel).write_bytes(audio.data)

            candidates: dict[str, Span] = {}
            for ref in seg.scene_refs:
                scene = scenes.get(ref)
                if scene is None:
                    raise PlanError(f"{seg.id} cites unknown scene {ref}")
                for sid in scene.shot_ids:
                    shot = shots.get(sid)
                    if shot is None:
                        raise PlanError(f"scene {ref} lists unknown shot {sid}")
                    candidates[sid] = Span(sid, shot.start_ms, shot.end_ms)
            clips = fill_clips(order_candidates(list(candidates.values()), used), audio.duration_ms)
            used.update(c.shot_id for c in clips)

            segments.append(
                PlanSegment(
                    id=seg.id,
                    kind="narration",
                    text_hash="sha256:" + hashlib.sha256(seg.text.encode("utf-8")).hexdigest(),
                    voice=voice,
                    audio=AudioRef(
                        file=rel,
                        duration_ms=audio.duration_ms,
                        char_timings=audio.char_timings,
                    ),
                    clips=[
                        Clip(
                            asset_id=asset_id,
                            shot_id=c.shot_id,
                            src_in_ms=c.src_in_ms,
                            src_out_ms=c.src_out_ms,
                        )
                        for c in clips
                    ],
                    source_audio=SourceAudio(
                        mode="duck", stem="mix", gain_db=NARRATION_SOURCE_GAIN_DB
                    ),
                )
            )
            ctx.progress((i + 1) / n, seg.id)

        plan = EditPlan(
            id=f"pln_{asset_id.split('_', 1)[1]}",
            project_id=script.project_id,
            version=1,
            author="ai",
            script_ref=ScriptRef(id=script.id, version=script.version),
            segments=segments,
        )
        write_model(ctx.out_dir / PLAN_FILE, plan)
        total_ms = sum(s.audio.duration_ms for s in segments if s.audio)
        return StageOutput(
            meta={
                "segments": len(segments),
                "clips": sum(len(s.clips) for s in segments),
                "duration_ms": total_ms,
            }
        )
