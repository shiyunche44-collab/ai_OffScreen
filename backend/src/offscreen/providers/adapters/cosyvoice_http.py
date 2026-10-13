"""CosyVoice sidecar over HTTP (`sidecars/cosyvoice/`): local synthesis that clones a voice
from a reference recording.

A voice with a reference recording in the library is cloned from it (with its transcript when
there is one, else cross-lingually); any other voice id is taken as a built-in speaker of the
loaded model. The recording is sent with every request, so the sidecar holds no state, and its
digest is part of the cache key (`voice_fingerprint`): a new recording is a new voice.

The sidecar is local, so nothing is retried: when it is down or fails, the job fails with what
to do about it. It gives no timestamps (`char_timings` is empty; M6-05 aligns them)."""

from __future__ import annotations

import base64
import hashlib
import io
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from offscreen.config import AppConfig
from offscreen.providers.ports import SynthesizedAudio, TTSError

SPEED_RANGE = (0.5, 2.0)
MAX_TEXT_CHARS = 10_000
SETUP_HINT = "start it as sidecars/cosyvoice/README.md describes"


@dataclass(frozen=True)
class VoiceReference:
    audio: bytes
    """A WAV recording of the voice."""
    text: str | None
    """What is said in it."""

    def fingerprint(self) -> str:
        h = hashlib.sha256(self.audio)
        h.update(b"\0" + (self.text or "").encode("utf-8"))
        return h.hexdigest()[:16]


References = Callable[[str], VoiceReference | None]
"""Voice id → its reference recording, or None for a built-in speaker."""


class CosyVoiceTTS:
    """Implements the `TTS` port with the CosyVoice sidecar at `cfg.sidecars.cosyvoice.url`."""

    def __init__(
        self,
        cfg: AppConfig,
        references: References,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.scfg = cfg.sidecars.cosyvoice
        self.references = references
        self._checked = False
        self._http = httpx.Client(
            base_url=self.scfg.url.rstrip("/"),
            timeout=httpx.Timeout(self.scfg.timeout_s, connect=5.0),
            transport=transport,
        )

    @property
    def id(self) -> str:
        return f"cosyvoice:{self.scfg.model}:wav"

    def close(self) -> None:
        self._http.close()

    def voice_fingerprint(self, voice_id: str) -> str:
        """Changes when the voice's recording or its transcript does (see `CachedTTS`)."""
        ref = self.references(voice_id)
        return ref.fingerprint() if ref else "builtin"

    def synthesize(self, text: str, *, voice_id: str, speed: float = 1.0) -> SynthesizedAudio:
        if not text.strip():
            raise ValueError("cannot synthesize empty text")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text is {len(text)} chars; the limit is {MAX_TEXT_CHARS}")
        if not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
            raise ValueError(f"speed {speed} outside {SPEED_RANGE}")
        self._check_model()
        body: dict[str, Any] = {"text": text, "speed": speed}
        ref = self.references(voice_id)
        if ref is None:
            body["speaker"] = voice_id
        else:
            body["prompt_audio"] = base64.b64encode(ref.audio).decode("ascii")
            if ref.text:
                body["prompt_text"] = ref.text
        resp = self._request("POST", "/tts", json=body)
        return _audio(resp.content, voice_id)

    def _check_model(self) -> None:
        """Once: the sidecar must run the model this engine's id names, or what it speaks
        would be cached under another model's name."""
        if self._checked:
            return
        health = self._request("GET", "/health").json()
        loaded = health.get("model") if isinstance(health, dict) else None
        if loaded != self.scfg.model:
            raise TTSError(
                f"cosyvoice sidecar runs {loaded!r} but sidecars.cosyvoice.model is "
                f"{self.scfg.model!r}; change one of them"
            )
        self._checked = True

    def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            resp = self._http.request(method, path, **kw)
        except httpx.ConnectError as e:
            raise TTSError(
                f"cosyvoice sidecar is not reachable at {self.scfg.url}; {SETUP_HINT}"
            ) from e
        except httpx.TimeoutException as e:
            raise TTSError(
                f"cosyvoice sidecar took longer than {self.scfg.timeout_s:g} s "
                "(sidecars.cosyvoice.timeout_s)"
            ) from e
        except httpx.TransportError as e:
            raise TTSError(f"cosyvoice sidecar: network error {type(e).__name__}") from e
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail")
            except ValueError:
                detail = None
            raise TTSError(
                f"cosyvoice sidecar: HTTP {resp.status_code}: {detail or resp.text[:300]}"
            )
        return resp


def _audio(data: bytes, voice_id: str) -> SynthesizedAudio:
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate, frames = w.getframerate(), w.getnframes()
    except (wave.Error, EOFError) as e:
        raise TTSError(f"cosyvoice sidecar returned no readable WAV for {voice_id}") from e
    duration_ms = round(frames * 1000 / rate) if rate else 0
    if duration_ms <= 0:
        raise TTSError(f"cosyvoice sidecar returned empty audio for {voice_id}")
    return SynthesizedAudio(
        data=data, format="wav", sample_rate=rate, duration_ms=duration_ms, char_timings=[]
    )
