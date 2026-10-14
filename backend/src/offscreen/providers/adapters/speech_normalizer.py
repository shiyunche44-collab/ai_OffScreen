"""Wraps any TTS engine so it speaks the normalized text (numbers, units, acronyms…) while
callers keep working with the script text: the returned character timings are mapped back to
the characters of the text that was passed in."""

from __future__ import annotations

from dataclasses import replace

from offscreen.algo.speech_text import NORMALIZER_VERSION, normalize_for_speech, remap_timings
from offscreen.providers.ports import TTS, SynthesizedAudio


class NormalizingTTS:
    def __init__(self, inner: TTS) -> None:
        self.inner = inner

    @property
    def id(self) -> str:
        return f"{self.inner.id}+norm{NORMALIZER_VERSION}"

    def synthesize(self, text: str, *, voice_id: str, speed: float = 1.0) -> SynthesizedAudio:
        st = normalize_for_speech(text)
        audio = self.inner.synthesize(st.spoken, voice_id=voice_id, speed=speed)
        if st.spoken == text:
            return audio
        return replace(audio, char_timings=remap_timings(st.pieces, audio.char_timings, len(text)))
