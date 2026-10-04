"""transcript: dialogue as timed lines (ARCHITECTURE §7.1).

An external subtitle file next to the movie wins: it is free, exact and already edited by
a human. Otherwise the ASR port recognizes the 16 kHz audio from the proxy stage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from offscreen.algo.subtitles import Cue, SubtitleParseError, guess_language, parse_subtitles
from offscreen.domain.index import Transcript, TranscriptLine
from offscreen.domain.job import Lane
from offscreen.engine import ArtifactRef, Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.providers.ports import ASR, AsrCanceled
from offscreen.stages.analysis.proxy import AUDIO_16K
from offscreen.store.files import write_model
from offscreen.store.repos import AssetRepo

TRANSCRIPT_FILE = "transcript.json"
SUBTITLE_SOURCE = "subtitle:external"


class TranscriptError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Subtitles:
    cues: list[Cue]
    sha256: str


def decode_subtitle_bytes(data: bytes) -> str:
    """UTF-16/UTF-8 with BOM, plain UTF-8, then GB18030 (the usual encoding of older
    Chinese subtitle files; it is a superset of GBK)."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("gb18030", errors="replace")


class TranscriptStage(Stage):
    name = "analysis.transcript"
    version = 1
    lane: Lane = "gpu"

    def __init__(self, assets: AssetRepo, asr: ASR | None = None) -> None:
        self.assets = assets
        self.asr = asr

    def inputs(self, scope: Scope) -> list[ArtifactRef]:
        return [ArtifactRef("analysis.proxy", scope)]

    def _subtitles(self, scope: Scope) -> _Subtitles | None:
        asset = self.assets.get(scope["asset_id"])
        if asset is None:
            raise TranscriptError(f"unknown asset {scope['asset_id']}")
        if not asset.subtitles_external:
            return None
        path = Path(asset.subtitles_external)
        if not path.is_file():
            return None  # recorded at import time but gone since
        data = path.read_bytes()
        try:
            cues = parse_subtitles(decode_subtitle_bytes(data), path.suffix)
        except SubtitleParseError as e:
            raise TranscriptError(f"cannot parse {path.name}: {e}") from e
        if not cues:
            return None  # an empty file carries no dialogue; fall back to ASR
        return _Subtitles(cues, hashlib.sha256(data).hexdigest())

    def params(self, scope: Scope) -> dict[str, Any]:
        subs = self._subtitles(scope)
        # The subtitle file is outside the artifact graph, so its content is in the key.
        if subs is not None:
            return {"asset_id": scope["asset_id"], "subtitle_sha256": subs.sha256}
        return {"asset_id": scope["asset_id"], "asr": self.asr.id if self.asr else None}

    def run(self, ctx: StageContext) -> StageOutput:
        asset_id = ctx.scope["asset_id"]
        subs = self._subtitles(ctx.scope)
        if subs is not None:
            doc = Transcript(
                asset_id=asset_id,
                language=guess_language(subs.cues),
                source=SUBTITLE_SOURCE,
                lines=[
                    TranscriptLine(
                        id=f"ln_{i:04d}", start_ms=c.start_ms, end_ms=c.end_ms, text=c.text
                    )
                    for i, c in enumerate(subs.cues, 1)
                ],
            )
        else:
            doc = self._recognize(ctx, asset_id)
        write_model(ctx.out_dir / TRANSCRIPT_FILE, doc)
        ctx.progress(1.0, "done")
        return StageOutput(meta={"lines": len(doc.lines), "source": doc.source})

    def _recognize(self, ctx: StageContext, asset_id: str) -> Transcript:
        proxy = ctx.input("analysis.proxy")
        if not proxy.meta.get("has_audio"):
            return Transcript(asset_id=asset_id, language="und", source="none", lines=[])
        if self.asr is None:
            raise TranscriptError(
                "no external subtitle and no ASR engine configured; "
                "place a .srt/.ass next to the movie or install faster-whisper"
            )
        try:
            result = self.asr.transcribe(
                proxy.path(AUDIO_16K),
                on_progress=lambda f: ctx.progress(f * 0.95, "transcribing"),
                should_cancel=ctx.is_canceled,
            )
        except AsrCanceled as e:
            raise StageCanceled(self.name) from e
        segs = sorted(result.segments, key=lambda s: (s.start_ms, s.end_ms))
        return Transcript(
            asset_id=asset_id,
            language=result.language,
            source=f"asr:{self.asr.id}",
            lines=[
                TranscriptLine(
                    id=f"ln_{i:04d}",
                    start_ms=s.start_ms,
                    end_ms=s.end_ms,
                    text=s.text,
                    words=s.words,
                )
                for i, s in enumerate(segs, 1)
            ],
        )
