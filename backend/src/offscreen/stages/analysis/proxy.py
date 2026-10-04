"""proxy: 540p browser proxy + 16 kHz mono and 48 kHz stereo wav (ARCHITECTURE §7.1)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from offscreen.domain.asset import MediaAsset
from offscreen.domain.job import Lane
from offscreen.engine import Scope, Stage, StageCanceled, StageContext, StageOutput
from offscreen.media.ffmpeg import FFmpegCanceled
from offscreen.media.transcode import PROXY_HEIGHT, extract_audio, make_proxy
from offscreen.store.repos import AssetRepo

PROXY_FILE = "proxy_540p.mp4"
AUDIO_16K = "audio_16k.wav"  # mono, for ASR
AUDIO_48K = "audio_48k.wav"  # stereo, for mixing and stem separation


class ProxyError(RuntimeError):
    pass


class ProxyStage(Stage):
    name = "analysis.proxy"
    version = 1
    lane: Lane = "cpu"

    # Share of the progress bar for each ffmpeg pass (the video encode dominates).
    _WEIGHTS = (0.8, 0.1, 0.1)

    def __init__(self, assets: AssetRepo) -> None:
        self.assets = assets

    def _asset(self, scope: Scope) -> MediaAsset:
        asset = self.assets.get(scope["asset_id"])
        if asset is None:
            raise ProxyError(f"unknown asset {scope['asset_id']}")
        return asset

    def params(self, scope: Scope) -> dict[str, Any]:
        a = self._asset(scope)
        # The source is outside the artifact graph, so its identity goes into the key.
        return {
            "asset_id": a.id,
            "fingerprint": a.fingerprint,
            "audio_stream": a.audio[0].index if a.audio else None,
            "height": PROXY_HEIGHT,
        }

    def run(self, ctx: StageContext) -> StageOutput:
        a = self._asset(ctx.scope)
        src = Path(a.source_path)
        if not src.is_file():
            raise ProxyError(f"source file is missing: {src}")
        stream = a.audio[0].index if a.audio else None

        # (output file, ffmpeg pass); one progress segment per pass.
        passes: list[tuple[str, Callable[[Path, Callable[[float], None]], None]]] = []

        def video(dst: Path, cb: Callable[[float], None]) -> None:
            make_proxy(
                src,
                dst,
                fps=a.video.fps,
                duration_ms=a.duration_ms,
                has_audio=stream is not None,
                on_progress=cb,
                should_cancel=ctx.is_canceled,
            )

        def audio(rate: int, channels: int) -> Callable[[Path, Callable[[float], None]], None]:
            def go(dst: Path, cb: Callable[[float], None]) -> None:
                assert stream is not None
                extract_audio(
                    src,
                    dst,
                    stream_index=stream,
                    sample_rate=rate,
                    channels=channels,
                    duration_ms=a.duration_ms,
                    on_progress=cb,
                    should_cancel=ctx.is_canceled,
                )

            return go

        passes.append((PROXY_FILE, video))
        if stream is not None:
            passes.append((AUDIO_16K, audio(16000, 1)))
            passes.append((AUDIO_48K, audio(48000, 2)))

        def reporter(label: str, start: float, share: float) -> Callable[[float], None]:
            return lambda f: ctx.progress(start + f * share, label)

        start = 0.0
        try:
            for (fname, go), share in zip(passes, self._WEIGHTS, strict=False):
                go(ctx.out_dir / fname, reporter(fname, start, share))
                start += share
        except FFmpegCanceled as e:
            raise StageCanceled(self.name) from e
        ctx.progress(1.0, "done")
        return StageOutput(meta={"has_audio": stream is not None})
