"""The voice library: register voices, measure how fast they speak.

Calibration speaks the standard texts at speed 1.0 and keeps the measured rate; the script is
then sized for the voice that will read it (`algo.calibrate.voice_timing`, used by the
pipeline). It costs a few hundred characters of synthesis and is cached like any other."""

from __future__ import annotations

from datetime import UTC, datetime

from offscreen.algo.calibrate import (
    STANDARD_TEXTS,
    TARGET_SPREAD,
    fit_rate,
    sample_of,
)
from offscreen.config import AppConfig
from offscreen.domain.voice import Voice, VoiceLibrary
from offscreen.providers.ports import TTSError
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.jobs import JobService
from offscreen.store.db import Database
from offscreen.store.voices import VoiceStore


class VoiceService:
    def __init__(self, cfg: AppConfig, db: Database, jobs: JobService) -> None:
        self.cfg = cfg
        self.db = db
        self.jobs = jobs
        self.store = VoiceStore(cfg.data_dir)

    def list(self) -> list[Voice]:
        return list(self.store.read().voices)

    def get(self, voice_id: str) -> Voice:
        found = next((v for v in self.store.read().voices if v.id == voice_id), None)
        if found is None:
            raise NotFound(f"voice {voice_id} is not in the library")
        return found

    def add(
        self,
        voice_id: str,
        *,
        name: str | None = None,
        provider: str | None = None,
        reference_audio: str | None = None,
        default_speed: float | None = None,
    ) -> Voice:
        """Register a voice, or change the fields given of a registered one. A changed
        provider or reference recording drops the measured rate (it described another voice)."""
        library = self.store.read()
        old = next((v for v in library.voices if v.id == voice_id), None)
        if reference_audio is not None and not (self.store.dir / reference_audio).is_file():
            raise InvalidInput(f"no recording {reference_audio} in {self.store.dir}")
        try:
            if old is None:
                voice = Voice(
                    id=voice_id,
                    name=name or voice_id,
                    provider=provider or self.cfg.tts.provider,
                    reference_audio=reference_audio,
                    default_speed=default_speed if default_speed is not None else 1.0,
                )
            else:
                changes: dict[str, object] = {
                    k: v
                    for k, v in (
                        ("name", name),
                        ("provider", provider),
                        ("reference_audio", reference_audio),
                        ("default_speed", default_speed),
                    )
                    if v is not None
                }
                if (
                    changes.get("provider", old.provider) != old.provider
                    or changes.get("reference_audio", old.reference_audio) != old.reference_audio
                ):
                    changes |= {
                        "chars_per_s": None,
                        "rate_spread": None,
                        "calibrated_with": None,
                        "calibrated_at": None,
                    }
                voice = Voice.model_validate(old.model_dump() | changes)
        except ValueError as e:
            raise InvalidInput(str(e)) from e
        self._save(library, voice)
        return voice

    def remove(self, voice_id: str) -> None:
        library = self.store.read()
        if all(v.id != voice_id for v in library.voices):
            raise NotFound(f"voice {voice_id} is not in the library")
        self.store.write(
            library.model_copy(update={"voices": [v for v in library.voices if v.id != voice_id]})
        )

    def calibrate(self, voice_id: str) -> Voice:
        """Measure the voice's speaking rate (registering the voice if it is not in the
        library yet) and keep it. Spends a few hundred characters of the TTS quota."""
        library = self.store.read()
        voice = next((v for v in library.voices if v.id == voice_id), None)
        if voice is None:
            voice = Voice(id=voice_id, name=voice_id, provider=self.cfg.tts.provider)
        tts = self.jobs.providers.tts
        samples = []
        try:
            for text in STANDARD_TEXTS:
                audio = tts.synthesize(text, voice_id=voice_id, speed=1.0)
                samples.append(sample_of(text, audio.duration_ms))
        except TTSError as e:
            raise InvalidInput(f"could not speak the standard texts: {e}") from e
        fit = fit_rate(samples)
        measured = voice.model_copy(
            update={
                "chars_per_s": round(fit.chars_per_s, 4),
                "rate_spread": round(fit.spread, 4),
                "calibrated_with": tts.id,
                "calibrated_at": datetime.now(UTC),
            }
        )
        self._save(library, measured)
        return measured

    @staticmethod
    def accurate(voice: Voice) -> bool:
        """The measured rate predicts the sample texts' lengths within `TARGET_SPREAD`."""
        return voice.rate_spread is not None and voice.rate_spread <= TARGET_SPREAD

    def _save(self, library: VoiceLibrary, voice: Voice) -> None:
        rest = [v for v in library.voices if v.id != voice.id]
        position = next((i for i, v in enumerate(library.voices) if v.id == voice.id), len(rest))
        rest.insert(position, voice)
        self.store.write(library.model_copy(update={"voices": rest}))
