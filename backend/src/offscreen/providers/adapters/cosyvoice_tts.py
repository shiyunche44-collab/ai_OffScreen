"""CosyVoice through its sidecar `POST {url}/synthesize` (sidecars/cosyvoice/README.md).

The way to clone a voice from a recording: MiniMax's Token Plan does not include voice cloning
(docs/PROVIDERS.md §2.7). A voice with a reference recording in the voice library is cloned
from it (with the recording's transcript when it is known: better prosody); any other voice id
is taken as one of the model's built-in speakers. The recording travels with every request, so
the sidecar keeps no state.

The engine gives no timestamps: `char_timings` is empty (word alignment is M6-05). The sidecar
runs locally, so a failure is not retried: either it is not running, or the request is wrong."""

from __future__ import annotations

import base64
import hashlib
import io
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from offscreen.config import AppConfig
from offscreen.providers.ports import SynthesizedAudio, TTSError

AUDIO_FORMAT = "wav"
API_VERSION = 1
SPEED_RANGE = (0.5, 2.0)
MAX_TEXT_CHARS = 2_000


@dataclass(frozen=True)
class VoicePrompt:
    """What a voice is cloned from."""

    audio: Path
    text: str | None = None
    """What is said in `audio`, if known."""


PromptLookup = Callable[[str], VoicePrompt | None]
"""voice id -> its reference recording, or None for a built-in speaker."""


class CosyVoiceTTS:
    """Implements the `TTS` port with the CosyVoice sidecar."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        prompts: PromptLookup,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = cfg.tts.model
        self.prompts = prompts
        c = cfg.tts.cosyvoice
        self._http = httpx.Client(
            base_url=c.url.rstrip("/"),
            timeout=httpx.Timeout(c.timeout_s, connect=5.0),
            transport=transport,
        )
        self._lock = threading.Lock()  # the sidecar does one synthesis at a time anyway
        self._checked = False

    @property
    def id(self) -> str:
        return f"cosyvoice:{self.model}:{AUDIO_FORMAT}"

    def close(self) -> None:
        self._http.close()

    def voice_key(self, voice_id: str) -> str:
        """Changes when the recording (or its transcript) a voice is cloned from changes:
        the same voice id then speaks differently (part of the TTS cache key)."""
        prompt = self.prompts(voice_id)
        if prompt is None:
            return ""
        h = hashlib.sha256(self._read(prompt))
        h.update(b"\0" + (prompt.text or "").encode("utf-8"))
        return h.hexdigest()[:16]

    def synthesize(self, text: str, *, voice_id: str, speed: float = 1.0) -> SynthesizedAudio:
        if not text.strip():
            raise ValueError("cannot synthesize empty text")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text is {len(text)} chars; the limit is {MAX_TEXT_CHARS}")
        if not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
            raise ValueError(f"speed {speed} is outside {SPEED_RANGE}")
        body: dict[str, Any] = {"text": text, "speed": speed}
        prompt = self.prompts(voice_id)
        if prompt is None:
            body["speaker"] = voice_id
        else:
            body["prompt_audio"] = base64.b64encode(self._read(prompt)).decode("ascii")
            if prompt.text:
                body["prompt_text"] = prompt.text
        with self._lock:
            self._check_model()
            resp = self._request("POST", "/synthesize", json=body)
        return _audio(resp.content)

    # ---- pieces ------------------------------------------------------------------------
    def _check_model(self) -> None:
        """Once: the sidecar answers and serves the configured model."""
        if self._checked:
            return
        health = self._request("GET", "/health").json()
        if health.get("api_version") != API_VERSION:
            raise TTSError(
                f"the CosyVoice sidecar speaks API {health.get('api_version')}, "
                f"expected {API_VERSION}: update sidecars/cosyvoice"
            )
        if health.get("model") != self.model:
            raise TTSError(
                f"the CosyVoice sidecar serves {health.get('model')!r} but tts.model is "
                f"{self.model!r}; set tts.model to the model it serves"
            )
        self._checked = True

    def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        url = self.cfg.tts.cosyvoice.url
        try:
            resp = self._http.request(method, path, **kw)
        except httpx.TimeoutException as e:
            raise TTSError(f"the CosyVoice sidecar at {url} did not answer in time") from e
        except httpx.HTTPError as e:
            raise TTSError(
                f"the CosyVoice sidecar at {url} is not reachable ({type(e).__name__}); "
                "start it as sidecars/cosyvoice/README.md describes"
            ) from e
        if resp.status_code != 200:
            try:
                detail = str(resp.json().get("error"))
            except ValueError:
                detail = resp.text
            raise TTSError(f"CosyVoice sidecar HTTP {resp.status_code}: {detail[:300]}")
        return resp

    @staticmethod
    def _read(prompt: VoicePrompt) -> bytes:
        try:
            return prompt.audio.read_bytes()
        except OSError as e:
            raise TTSError(f"cannot read the reference recording {prompt.audio}: {e}") from e


def _audio(data: bytes) -> SynthesizedAudio:
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            rate, frames = w.getframerate(), w.getnframes()
    except (wave.Error, EOFError) as e:
        raise TTSError("the CosyVoice sidecar did not return a WAV file") from e
    if frames <= 0 or rate <= 0:
        raise TTSError("the CosyVoice sidecar returned empty audio")
    return SynthesizedAudio(
        data=data,
        format=AUDIO_FORMAT,
        sample_rate=rate,
        duration_ms=max(1, round(frames * 1000 / rate)),
        char_timings=[],
    )
