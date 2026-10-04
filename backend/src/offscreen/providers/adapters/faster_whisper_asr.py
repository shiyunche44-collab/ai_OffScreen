"""ASR backed by faster-whisper (local, word timestamps). Install the `gpu` extra to use it;
the package is imported lazily so everything else works without it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from offscreen.domain.index import Word
from offscreen.providers.ports import AsrCanceled, AsrResult, AsrSegment

# Raise `VERSION` when the settings below change: it is part of the cache key.
VERSION = 1
BEAM_SIZE = 5


def _ms(seconds: float) -> int:
    return max(0, round(seconds * 1000))


class FasterWhisperAsr:
    def __init__(
        self,
        model: str = "large-v3",
        *,
        device: str = "cuda",
        compute_type: str | None = None,
        engine: Any = None,
    ) -> None:
        """`engine` is a ready WhisperModel-like object (tests inject a fake)."""
        self.model = model
        self.device = device
        self.compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self._engine = engine

    @property
    def id(self) -> str:
        # device and compute type are left out on purpose: they change numerics only
        # marginally and moving between machines should not invalidate the cache.
        return f"faster-whisper/{self.model}@{VERSION}"

    def _load(self) -> Any:
        if self._engine is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise RuntimeError(
                    "faster-whisper is not installed; install it (the `gpu` extra) "
                    "or provide an external subtitle file"
                ) from e
            self._engine = WhisperModel(
                self.model, device=self.device, compute_type=self.compute_type
            )
        return self._engine

    def transcribe(
        self,
        wav: Path,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AsrResult:
        segments, info = self._load().transcribe(
            str(wav),
            language=language,
            beam_size=BEAM_SIZE,
            word_timestamps=True,
            vad_filter=True,
        )
        total = float(getattr(info, "duration", 0.0) or 0.0)
        out: list[AsrSegment] = []
        for seg in segments:  # lazy generator: the actual decoding happens here
            if should_cancel is not None and should_cancel():
                raise AsrCanceled("transcription canceled")
            text = seg.text.strip()
            start = _ms(seg.start)
            end = max(_ms(seg.end), start + 1)
            if text:
                words = [
                    Word(
                        w=w.word.strip(),
                        start_ms=_ms(w.start),
                        end_ms=max(_ms(w.end), _ms(w.start) + 1),
                    )
                    for w in (seg.words or [])
                    if w.word.strip()
                ]
                out.append(AsrSegment(start, end, text, words))
            if on_progress is not None and total > 0:
                on_progress(min(1.0, seg.end / total))
        return AsrResult(language=info.language, segments=out)
