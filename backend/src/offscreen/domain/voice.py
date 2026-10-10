"""The voice library: the voices a person can narrate with, and how fast each one speaks."""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from offscreen.domain.common import Strict, Versioned


class Voice(Strict):
    id: str = Field(min_length=1)
    """The id the TTS engine knows the voice by (what `PlanSegment.voice` carries)."""
    name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    """The TTS engine the voice belongs to (`tts.provider`)."""
    reference_audio: str | None = None
    """A recording to clone the voice from, relative to `library/voices/`."""
    default_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    """The speed plans use for this voice unless a person sets another."""
    chars_per_s: float | None = Field(default=None, gt=0)
    """Measured speaking rate at speed 1.0: spoken characters per second of audio."""
    rate_spread: float | None = Field(default=None, ge=0)
    """How far the sample texts strayed from `chars_per_s` (relative; 0.03 is 3 %): the error
    to expect when estimating a text's length."""
    calibrated_with: str | None = None
    """`TTS.id` of the engine that was measured; another engine's rate is not trusted."""
    calibrated_at: datetime | None = None

    @model_validator(mode="after")
    def _calibration_is_whole(self) -> Self:
        parts = (self.chars_per_s, self.rate_spread, self.calibrated_with, self.calibrated_at)
        if any(p is None for p in parts) and any(p is not None for p in parts):
            raise ValueError("calibration needs chars_per_s, rate_spread, calibrated_with and _at")
        return self


class VoiceLibrary(Versioned):
    voices: list[Voice] = []

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [v.id for v in self.voices]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate voice ids")
        return self
