"""Wiring: the configured TTS engine behind a disk cache under `data_dir/tts_cache`.

`tts.provider: cosyvoice` speaks through the CosyVoice sidecar, cloning the voices of the
voice library from their reference recordings; any other provider is MiniMax-style."""

from __future__ import annotations

from offscreen.config import AppConfig
from offscreen.providers.adapters.cosyvoice_http import (
    CosyVoiceTTS,
    References,
    VoiceReference,
)
from offscreen.providers.adapters.minimax_tts import MiniMaxTTS
from offscreen.providers.adapters.tts_cache import CachedTTS
from offscreen.providers.ports import TTS, TTSError
from offscreen.store.voices import VoiceStore


def build_tts(cfg: AppConfig) -> CachedTTS:
    inner: TTS
    if cfg.tts.provider == "cosyvoice":
        inner = CosyVoiceTTS(cfg, library_references(VoiceStore(cfg.data_dir)))
    else:
        inner = MiniMaxTTS(cfg)
    return CachedTTS(inner, cfg.data_dir / "tts_cache")


def library_references(store: VoiceStore) -> References:
    """A library voice with a recording is cloned from it; other ids are built-in speakers."""

    def reference(voice_id: str) -> VoiceReference | None:
        voice = store.get(voice_id)
        path = store.reference_path(voice) if voice else None
        if voice is None or path is None:
            return None
        try:
            audio = path.read_bytes()
        except OSError as e:
            raise TTSError(f"voice {voice_id}: cannot read its recording {path}: {e}") from e
        return VoiceReference(audio=audio, text=voice.reference_text)

    return reference
