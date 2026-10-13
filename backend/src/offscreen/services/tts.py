"""Wiring: the configured TTS engine behind a disk cache under `data_dir/tts_cache`."""

from __future__ import annotations

from offscreen.config import AppConfig
from offscreen.providers.adapters.cosyvoice_tts import CosyVoiceTTS, PromptLookup, VoicePrompt
from offscreen.providers.adapters.minimax_tts import MiniMaxTTS
from offscreen.providers.adapters.tts_cache import CachedTTS
from offscreen.providers.adapters.volc_tts import VolcTTS
from offscreen.store.voices import VoiceStore


def build_tts(cfg: AppConfig) -> CachedTTS:
    cache = cfg.data_dir / "tts_cache"
    if cfg.tts.provider == "cosyvoice":
        cosy = CosyVoiceTTS(cfg, prompts=library_prompts(cfg))
        return CachedTTS(cosy, cache, voice_key=cosy.voice_key)
    pcfg = cfg.providers.get(cfg.tts.provider)
    if pcfg is not None and pcfg.kind == "volc_tts":
        return CachedTTS(VolcTTS(cfg), cache)
    return CachedTTS(MiniMaxTTS(cfg), cache)


def library_prompts(cfg: AppConfig) -> PromptLookup:
    """A voice in the library with a reference recording is cloned from it."""
    store = VoiceStore(cfg.data_dir)

    def lookup(voice_id: str) -> VoicePrompt | None:
        voice = next((v for v in store.read().voices if v.id == voice_id), None)
        if voice is None or voice.reference_audio is None:
            return None
        return VoicePrompt(store.dir / voice.reference_audio, voice.reference_text)

    return lookup
