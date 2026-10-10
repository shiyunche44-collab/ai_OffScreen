"""plan: script + scenes + shots (+ shot vectors) -> plan.json, with the voice-over audio (T3).

Per narration segment: the TTS port speaks the text, and footage is chosen in three steps
(ARCHITECTURE §7.3): candidates (the shots of the cited scenes, the best matches of a vector
search, `algo.candidates`), a weighted score (`algo.scoring`, with a character-bigram text
similarity, which works without spaces), then the best-scored shots are cut and sped up or down
to fill the voice-over to the millisecond (`algo.fitting`). Original-sound segments take the
lines they cite, padded (`algo.original`), and narration is kept off that footage.

The plan's id follows the project's, so the first build and the document store's copy of it are
the same bytes; a build that changes nothing returns the plan it started from, untouched, so
the compile and render stages after it keep their caches when the plan has been stored as a
document and is the next build's starting point.

The build is incremental. `PlanSettings.previous` is the plan to start from (the project's
current version in the document store): a narration segment whose script text, voice and audio
are unchanged is carried over untouched, and only the others get new voice-over and footage;
locked clips survive a rebuild (`algo.plan_build`). `PlanSettings.script` is the script to
follow when a person has edited it (the document store's version); without it the stage reads
the generated one. Both are part of the cache key. What was rebuilt, and why, goes to the job
log and the artifact's manifest.

Audio files are written under `tts/` in this artifact and `AudioRef.file` is relative to the
directory of the plan document, as for every plan version."""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from offscreen.algo.candidates import candidates_for_segment
from offscreen.algo.fitting import MAX_CLIP_MS, MIN_CLIP_MS, MIN_SPEED, Shot, fit_duration
from offscreen.algo.original import find_overlaps, original_segment
from offscreen.algo.plan_build import (
    PlanReport,
    decide_narration,
    keep_locked,
    plan_order,
    same_content,
    text_digest,
    voice_for,
    with_fresh_footage,
)
from offscreen.algo.scoring import ScoringWeights, score_candidates
from offscreen.algo.search import search_text
from offscreen.algo.textsim import coverage
from offscreen.algo.tts import tts_cache_key
from offscreen.algo.vectors import TOP_K, rank_by_similarity, search_signal
from offscreen.domain.index import (
    Captions,
    Cast,
    Characters,
    Scene,
    Scenes,
    ShotCaption,
    ShotIndex,
    ShotIndexEntry,
    Shots,
    Transcript,
)
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
from offscreen.domain.script import Script, ScriptSegment
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.providers.ports import TTS, Embedder
from offscreen.stages.analysis.captions import CAPTIONS_FILE
from offscreen.stages.analysis.embeddings import (
    IMAGE_VECTORS_FILE,
    SHOT_INDEX_FILE,
    TEXT_VECTORS_FILE,
)
from offscreen.stages.analysis.scenes import SCENES_FILE
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.stages.analysis.transcript import TRANSCRIPT_FILE
from offscreen.stages.creation.script import SCRIPT_FILE
from offscreen.store.files import write_model

PLAN_FILE = "plan.json"
AUDIO_DIR = "tts"
NARRATION_SOURCE_GAIN_DB = -20.0
"""Film sound under the narration (ARCHITECTURE M1-13: original audio at -20 dB)."""
POOL = 12
"""How many of the best-scored shots a segment's footage is cut from (a 12 s segment needs
about six)."""
MIN_FILL_MS = round(MIN_CLIP_MS / 1.15) + 1
"""Shortest footage worth adding around locked clips (the shortest clip plays this long)."""
NEUTRAL_QUALITY = 0.5
"""Sharpness / brightness assumed for a shot without a measurement."""

log = logging.getLogger(__name__)


class PlanError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreviousPlan:
    plan: EditPlan
    directory: Path
    """Where the plan's audio files are (`AudioRef.file` is relative to it)."""


@dataclass(frozen=True)
class PlanSettings:
    script: Script | None = None
    """The script to follow; None: the generated one (`creation.script`)."""
    previous: PreviousPlan | None = None
    """The plan to build on; None: build everything."""
    previous_script_ids: tuple[str, ...] | None = None
    """Segment ids of the script `previous` was built from; with them, a person's structural
    edits of the plan (order, deleted segments, inserted original sound) are kept."""
    weights: ScoringWeights = field(default_factory=ScoringWeights)


class PlanStage(Stage):
    name = "creation.plan"
    version = 3  # 3: scored footage, speed fitting, incremental build, original segments
    lane: Lane = "api"

    def __init__(
        self,
        tts: TTS,
        voice_speed: float = 1.0,
        settings: PlanSettings | None = None,
        image_embedder: Embedder | None = None,
        text_embedder: Embedder | None = None,
    ) -> None:
        if not 0.5 <= voice_speed <= 2.0:
            raise ValueError("voice_speed must be within 0.5-2.0")
        self.tts = tts
        self.voice_speed = voice_speed
        self.settings = settings or PlanSettings()
        self.image_embedder = image_embedder
        self.text_embedder = text_embedder

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        refs = [
            ArtifactRef("analysis.scenes", scope),
            ArtifactRef("analysis.shots", scope),
            ArtifactRef("analysis.captions", scope),
            ArtifactRef("analysis.transcript", scope),
        ]
        if self.settings.script is None:
            refs.insert(0, ArtifactRef("creation.script", scope))
        if self.image_embedder is not None or self.text_embedder is not None:
            refs.append(ArtifactRef("analysis.embeddings", scope))
        return refs

    def params(self, scope: Scope) -> dict[str, Any]:
        s = self.settings
        return {
            "asset_id": scope["asset_id"],
            "voice_speed": self.voice_speed,
            "max_clip_ms": MAX_CLIP_MS,
            "min_clip_ms": MIN_CLIP_MS,
            "min_speed": MIN_SPEED,
            "pool": POOL,
            "top_k": TOP_K,
            "source_gain_db": NARRATION_SOURCE_GAIN_DB,
            "weights": s.weights.__dict__,
            "script": _digest(s.script.model_dump_json()) if s.script else None,
            "previous": _digest(s.previous.plan.model_dump_json()) if s.previous else None,
            "previous_script": list(s.previous_script_ids)
            if s.previous_script_ids is not None
            else None,
        }

    def provider_info(self, scope: Scope) -> dict[str, Any]:
        return {
            "tts": self.tts.id,
            "image_embedder": self.image_embedder.id if self.image_embedder else "none",
            "text_embedder": self.text_embedder.id if self.text_embedder else "none",
        }

    def run(self, ctx: StageContext) -> StageOutput:
        asset_id = ctx.scope["asset_id"]
        script = self.settings.script or ctx.input("creation.script").read_model(
            SCRIPT_FILE, Script
        )
        scenes = {
            s.id: s for s in ctx.input("analysis.scenes").read_model(SCENES_FILE, Scenes).scenes
        }
        shots = ctx.input("analysis.shots").read_model(SHOTS_FILE, Shots)
        captions = {
            c.shot_id: c
            for c in ctx.input("analysis.captions").read_model(CAPTIONS_FILE, Captions).captions
        }
        transcript = ctx.input("analysis.transcript").read_model(TRANSCRIPT_FILE, Transcript)
        previous = self.settings.previous
        before = {s.id: s for s in previous.plan.segments} if previous else {}
        voice = VoiceSpec(voice_id=script.params.voice_id, speed=self.voice_speed)
        (ctx.out_dir / AUDIO_DIR).mkdir()

        picker = _Picker(
            shots=shots,
            scenes=scenes,
            captions=captions,
            weights=self.settings.weights,
            search=self._search(ctx),
        )
        by_id = {seg.id: seg for seg in script.segments}
        order = plan_order(
            [seg.id for seg in script.segments],
            previous.plan if previous else None,
            self.settings.previous_script_ids,
        )
        report = PlanReport(removed=[sid for sid in before if sid not in order])

        # Original-sound segments first: narration must stay off their footage. A person's
        # trimming of one is kept while it still plays the same lines.
        done: dict[str, PlanSegment] = {}
        for sid in order:
            seg = by_id.get(sid)
            old = before.get(sid)
            if seg is None:  # inserted by a person
                assert old is not None
                done[sid] = old
            elif seg.kind == "original":
                keep = (
                    old is not None
                    and old.kind == "original"
                    and bool(old.line_refs)
                    and old.line_refs == seg.line_refs
                )
                done[sid] = old if keep and old is not None else _original(seg, transcript, shots)
        taken = [c for p in done.values() for c in p.clips]

        # Then decide which narration segments stay, so a rebuild avoids their shots.
        used: set[str] = set()
        rebuild: dict[str, ScriptSegment] = {}
        for sid in order:
            seg = by_id.get(sid)
            if seg is None or seg.kind != "narration":
                continue
            old = before.get(seg.id)
            decision = decide_narration(
                seg.id,
                seg.text,
                voice_for(old, voice),
                old,
                audio_exists=bool(
                    old
                    and old.audio
                    and previous
                    and (previous.directory / old.audio.file).is_file()
                ),
            )
            if decision.reuse and old is not None and previous is not None:
                done[seg.id] = self._carry_over(ctx, previous, old)
                used.update(c.shot_id for c in old.clips if c.shot_id)
                report.reused.append(seg.id)
            else:
                assert decision.reason is not None
                rebuild[seg.id] = seg
                report.rebuilt[seg.id] = decision.reason
                if old is not None:
                    used.update(c.shot_id for c in old.clips if c.locked and c.shot_id)

        n = max(1, len(rebuild))
        for i, seg in enumerate(rebuild.values()):
            if ctx.is_canceled():
                raise StageCanceled(self.name)
            old = before.get(seg.id)
            done[seg.id] = self._build_narration(
                ctx, seg, voice_for(old, voice), old, picker, used, taken, asset_id
            )
            ctx.progress((i + 1) / n, seg.id)

        segments = [done[sid] for sid in order]
        for o in find_overlaps(segments):
            report.warnings.append(
                f"{o.original_id} (original sound) and {o.narration_id} show the same footage "
                f"({o.start_ms}-{o.end_ms} ms)"
            )
        for line in report.lines():
            (log.warning if line.startswith("plan: warning") else log.info)("%s", line)

        prev = previous.plan if previous else None
        plan = EditPlan(
            id=f"pln_{script.project_id.split('_', 1)[1]}",
            project_id=script.project_id,
            version=prev.version + 1 if prev else 1,
            parent_version=prev.version if prev else None,
            author="ai",
            script_ref=ScriptRef(id=script.id, version=script.version),
            segments=segments,
            bgm=prev.bgm if prev else None,
            output_profile=prev.output_profile if prev else "source",
        )
        if prev is not None and same_content(plan, prev):
            plan = prev  # nothing to do: the very same plan, so what follows stays cached
        write_model(ctx.out_dir / PLAN_FILE, plan)
        total_ms = sum(_segment_ms(s) for s in segments)
        return StageOutput(
            meta={
                "segments": len(segments),
                "clips": sum(len(s.clips) for s in segments),
                "duration_ms": total_ms,
                "rebuilt": dict(report.rebuilt),
                "reused": report.reused,
                "removed": report.removed,
                "warnings": report.warnings,
            }
        )

    # ---- pieces ------------------------------------------------------------------------
    def _carry_over(
        self, ctx: StageContext, previous: PreviousPlan, old: PlanSegment
    ) -> PlanSegment:
        assert old.audio is not None
        dst = ctx.out_dir / old.audio.file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous.directory / old.audio.file, dst)
        return old

    def _build_narration(
        self,
        ctx: StageContext,
        seg: ScriptSegment,
        voice: VoiceSpec,
        old: PlanSegment | None,
        picker: _Picker,
        used: set[str],
        taken: list[Clip],
        asset_id: str,
    ) -> PlanSegment:
        audio = self.tts.synthesize(seg.text, voice_id=voice.voice_id, speed=voice.speed)
        key = tts_cache_key(seg.text, voice.voice_id, voice.speed, self.tts.id)
        rel = f"{AUDIO_DIR}/{key.removeprefix('sha256:')[:16]}.{audio.format}"
        (ctx.out_dir / rel).write_bytes(audio.data)

        kept = keep_locked(old.clips if old and old.kind == "narration" else [], audio.duration_ms)
        fresh: list[Clip] = []
        if kept.remaining_ms > 0:
            fresh = picker.pick(seg, max(kept.remaining_ms, MIN_FILL_MS), used, taken, asset_id)
        clips = with_fresh_footage(kept, fresh)
        used.update(c.shot_id for c in clips if c.shot_id)
        return PlanSegment(
            id=seg.id,
            kind="narration",
            text=seg.text,
            text_hash=text_digest(seg.text),
            voice=voice,
            voice_pinned=bool(old and old.kind == "narration" and old.voice_pinned),
            audio=AudioRef(
                file=rel, duration_ms=audio.duration_ms, char_timings=audio.char_timings
            ),
            clips=clips,
            source_audio=SourceAudio(mode="duck", stem="mix", gain_db=NARRATION_SOURCE_GAIN_DB),
        )

    def _search(self, ctx: StageContext) -> _Search | None:
        if self.image_embedder is None and self.text_embedder is None:
            return None
        art = ctx.input("analysis.embeddings")
        index = art.read_model(SHOT_INDEX_FILE, ShotIndex)
        ids = [e.shot_id for e in index.shots]
        columns: list[tuple[str, Embedder, Any]] = []
        if self.image_embedder is not None and index.image_model is not None:
            columns.append(("image", self.image_embedder, np.load(art.path(IMAGE_VECTORS_FILE))))
        if self.text_embedder is not None and index.text_model is not None:
            columns.append(("text", self.text_embedder, np.load(art.path(TEXT_VECTORS_FILE))))
        return _Search(ids, columns)


class _Search:
    """A segment's text compared with every shot, by picture and by description."""

    def __init__(self, ids: list[str], columns: list[tuple[str, Embedder, Any]]) -> None:
        self.ids = ids
        self.columns = columns

    def __call__(self, text: str) -> tuple[list[str], dict[str, float]]:
        ranked: dict[str, list[tuple[str, float]]] = {}
        for name, embedder, matrix in self.columns:
            (vector,) = embedder.embed_texts([text])
            try:
                ranked[name] = rank_by_similarity(self.ids, matrix, vector)
            except ValueError as e:
                raise PlanError(f"{name} vectors do not match the query embedder: {e}") from e
        return search_signal(ranked)


class _Picker:
    """Chooses and cuts the footage for one narration segment."""

    def __init__(
        self,
        *,
        shots: Shots,
        scenes: dict[str, Scene],
        captions: dict[str, ShotCaption],
        weights: ScoringWeights,
        search: _Search | None,
    ) -> None:
        self.shots = {s.id: s for s in shots.shots}
        self.shot_list = shots
        self.scenes = scenes
        self.scene_shots = {sid: list(sc.shot_ids) for sid, sc in scenes.items()}
        self.captions = captions
        self.weights = weights
        self.search = search
        scene_of = {sh: sc.id for sc in scenes.values() for sh in sc.shot_ids}
        self.entries = {
            s.id: ShotIndexEntry(
                shot_id=s.id,
                scene_id=scene_of.get(s.id),
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                sharpness=s.quality.sharpness if s.quality else NEUTRAL_QUALITY,
                brightness=s.quality.brightness if s.quality else NEUTRAL_QUALITY,
                is_credits=bool(captions[s.id].is_credits) if s.id in captions else False,
                caption=search_text(captions[s.id]) if s.id in captions else "",
            )
            for s in shots.shots
        }

    def pick(
        self,
        seg: ScriptSegment,
        target_ms: int,
        used: set[str],
        taken: list[Clip],
        asset_id: str,
    ) -> list[Clip]:
        for ref in seg.scene_refs:
            if ref not in self.scenes:
                raise PlanError(f"{seg.id} cites unknown scene {ref}")
        top, similarity = self.search(seg.text) if self.search else ([], {})
        cand = candidates_for_segment(
            seg,
            self.shot_list,
            self.scene_shots,
            self.captions,
            Cast(asset_id=asset_id, shots=[]),
            Characters(asset_id=asset_id, characters=[]),
            {"embedding": top} if top else None,
        )
        ids = [sid for sid in cand.shot_ids if self._usable(sid, taken)]
        if not ids:
            raise PlanError(f"{seg.id}: no usable footage among {len(cand.shot_ids)} candidates")
        starts = [self.scenes[ref].start_ms for ref in seg.scene_refs]
        scored = score_candidates(
            ids,
            self.entries,
            {},
            self.captions,
            seg.text,
            similarity,
            set(),
            used,
            min(starts),
            self.weights,
            caption_similarity=coverage,
        )[:POOL]
        total = {s.shot_id: s.total for s in scored}
        try:
            cuts = fit_duration(
                [
                    Shot(
                        s.shot_id,
                        self.shots[s.shot_id].end_ms - self.shots[s.shot_id].start_ms,
                        self.shots[s.shot_id].start_ms,
                    )
                    for s in scored
                ],
                target_ms,
            )
        except ValueError as e:
            raise PlanError(f"{seg.id}: {e}") from e
        cuts.sort(key=lambda c: c.src_in_ms)  # footage runs forward in film time
        return [
            Clip(
                asset_id=asset_id,
                shot_id=c.shot_id,
                src_in_ms=c.src_in_ms,
                src_out_ms=c.src_out_ms,
                speed=round(c.speed, 6),
                score=round(total[c.shot_id], 4),
            )
            for c in cuts
        ]

    def _usable(self, shot_id: str, taken: list[Clip]) -> bool:
        entry = self.entries.get(shot_id)
        if entry is None or entry.is_credits:
            return False
        return not any(t.src_in_ms < entry.end_ms and entry.start_ms < t.src_out_ms for t in taken)


def _original(seg: ScriptSegment, transcript: Transcript, shots: Shots) -> PlanSegment:
    try:
        return original_segment(
            seg.id, seg.line_refs, transcript, shots.shots[-1].end_ms if shots.shots else None
        )
    except ValueError as e:
        raise PlanError(f"{seg.id}: {e}") from e


def _segment_ms(seg: PlanSegment) -> int:
    if seg.audio is not None:
        return seg.audio.duration_ms
    return round(sum((c.src_out_ms - c.src_in_ms) / c.speed for c in seg.clips))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
