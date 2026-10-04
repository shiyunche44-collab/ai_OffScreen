"""Wiring: the configured TTS engine behind a disk cache under `data_dir/tts_cache`."""

from __future__ import annotations

from offscreen.config import AppConfig
from offscreen.providers.adapters.minimax_tts import MiniMaxTTS
from offscreen.providers.adapters.tts_cache import CachedTTS


def build_tts(cfg: AppConfig) -> CachedTTS:
    return CachedTTS(MiniMaxTTS(cfg), cfg.data_dir / "tts_cache")
