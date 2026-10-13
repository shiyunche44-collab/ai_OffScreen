"""Wiring: the configured TTS engine behind text normalization and a disk cache under
`data_dir/tts_cache`. The cache sits closest to the engine, so it is keyed by the normalized
text that was actually spoken."""

from __future__ import annotations

from dataclasses import replace

from offscreen.algo.normalize import normalize_for_speech, remap_timings
from offscreen.config import AppConfig
from offscreen.providers.adapters.minimax_tts import MiniMaxTTS
from offscreen.providers.adapters.tts_cache import CachedTTS
from offscreen.providers.ports import TTS, SynthesizedAudio


class NormalizingTTS:
    """Speaks numbers, units and symbols the way a narrator reads them; timestamps still refer
    to the characters of the text it was given."""

    def __init__(self, inner: TTS) -> None:
        self.inner = inner

    @property
    def id(self) -> str:
        return self.inner.id

    def synthesize(self, text: str, *, voice_id: str, speed: float = 1.0) -> SynthesizedAudio:
        norm = normalize_for_speech(text)
        result = self.inner.synthesize(norm.text, voice_id=voice_id, speed=speed)
        if norm.text == text:
            return result
        timings = remap_timings(norm, result.char_timings, len(text), result.duration_ms)
        return replace(result, char_timings=timings)


def build_tts(cfg: AppConfig) -> NormalizingTTS:
    return NormalizingTTS(CachedTTS(MiniMaxTTS(cfg), cfg.data_dir / "tts_cache"))
