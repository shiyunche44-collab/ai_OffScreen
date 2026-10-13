"""Volcengine Doubao speech synthesis, V3 HTTP `POST {base_url}/api/v3/tts/unidirectional`
(docs/PROVIDERS.md §5, measured 2026-10-13).

Auth is the console's API key in `X-Api-Key`; `tts.model` is the resource id (`seed-tts-2.0`;
a voice only works with the resource it belongs to). The response is newline-delimited JSON:
audio chunks (`data`, base64), `sentence` events whose `words` carry timestamps in seconds
when `audio_params.enable_subtitle` is set, and a last event with code 20000000. Words come as
text without offsets, so they are matched back onto the input (`algo.tts.spans_from_words`).

Audio is requested as raw PCM and wrapped in a WAV here: the engine's own WAV header carries
placeholder sizes (it is streamed). Errors arrive as HTTP 4xx with `header.code`, or as HTTP 200
with a top-level `code`."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
import uuid
import wave
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from offscreen.algo.tts import char_timings_from_spans, spans_from_words
from offscreen.config import AppConfig
from offscreen.providers.adapters.http_util import (
    MAX_RETRIES,
    FailureKind,
    backoff_delay,
    classify_failure,
    mentions_quota,
    parse_retry_after,
)
from offscreen.providers.ports import (
    SynthesizedAudio,
    TTSAuthError,
    TTSError,
    TTSQuotaExhausted,
    TTSRateLimited,
)

DEFAULT_BASE_URL = "https://openspeech.bytedance.com"
PATH = "/api/v3/tts/unidirectional"
TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=60.0)
AUDIO_FORMAT = "wav"
SPEED_RANGE = (0.5, 2.0)
MAX_TEXT_CHARS = 1_000
OK_DONE = 20_000_000

# `code` values seen (measured) and what they mean for us.
_AUTH_CODES = {45_000_000, 45_000_010, 45_000_030}  # no key, bad key, resource not granted
_FATAL_CODES = {55_000_000, 45_002_001}  # voice not in this resource; no readable text


class MalformedStream(TTSError):
    """The response did not have the measured shape."""


@dataclass
class ParsedStream:
    pcm: bytearray = field(default_factory=bytearray)
    words: list[tuple[str, float, float]] = field(default_factory=list)
    done: bool = False
    error: dict[str, Any] | None = None


def parse_stream(lines: Iterable[str]) -> ParsedStream:
    """Fold the response lines into PCM, word timings (ms) and whether it finished."""
    out = ParsedStream()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError as e:
            raise MalformedStream(f"unparseable event: {line[:200]}") from e
        code = _code(event)
        if code == OK_DONE:
            out.done = True
            continue
        if code not in (0, None):
            out.error = event
            return out
        if event.get("data"):
            try:
                out.pcm += base64.b64decode(event["data"], validate=True)
            except ValueError as e:
                raise MalformedStream("audio is not valid base64") from e
        sentence = event.get("sentence") or {}
        for w in sentence.get("words") or []:
            out.words.append(
                (str(w["word"]), float(w["startTime"]) * 1000, float(w["endTime"]) * 1000)
            )
    return out


def _code(event: Any) -> int | None:
    if not isinstance(event, dict):
        return None
    code = event.get("code")
    if code is None and isinstance(event.get("header"), dict):
        code = event["header"].get("code")
    return code if isinstance(code, int) else None


def _message(event: Any, text: str = "") -> str:
    if isinstance(event, dict):
        msg = event.get("message")
        if msg is None and isinstance(event.get("header"), dict):
            msg = event["header"].get("message")
        if msg:
            return f"{_code(event)} {msg}"[:300]
    return text[:300]


def speech_rate(speed: float) -> int:
    """The engine's `speech_rate` (-50..100) for a speed factor: measured, -50 speaks at about
    half speed, 0 normally, 100 twice as fast."""
    return max(-50, min(100, round((speed - 1.0) * 100)))


class VolcTTS:
    """Implements the `TTS` port with the provider named by `cfg.tts.provider`."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.provider = cfg.tts.provider
        pcfg = cfg.providers.get(self.provider)
        if pcfg is None or pcfg.kind != "volc_tts":
            raise ValueError(f"tts.provider {self.provider!r} is not a volc_tts provider")
        self._limiter = threading.BoundedSemaphore(pcfg.max_concurrency)
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=(pcfg.base_url or DEFAULT_BASE_URL).rstrip("/"),
            timeout=TIMEOUT,
            transport=transport,
        )

    @property
    def id(self) -> str:
        t = self.cfg.tts
        return f"{self.provider}:{t.model}:{AUDIO_FORMAT}@{t.sample_rate}"

    def close(self) -> None:
        self._http.close()

    def synthesize(self, text: str, *, voice_id: str, speed: float = 1.0) -> SynthesizedAudio:
        if not text.strip():
            raise ValueError("cannot synthesize empty text")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text is {len(text)} chars; the limit is {MAX_TEXT_CHARS}")
        if not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
            raise ValueError(f"speed {speed} outside {SPEED_RANGE}")
        t = self.cfg.tts
        payload = {
            "user": {"uid": "offscreen"},
            "req_params": {
                "text": text,
                "speaker": voice_id,
                "audio_params": {
                    "format": "pcm",
                    "sample_rate": t.sample_rate,
                    "speech_rate": speech_rate(speed),
                    "enable_subtitle": True,
                },
            },
        }
        parsed = self._post(payload)
        return self._result(text, parsed)

    def _result(self, text: str, parsed: ParsedStream) -> SynthesizedAudio:
        rate = self.cfg.tts.sample_rate
        frames = len(parsed.pcm) // 2
        if frames == 0:
            raise TTSError(f"{self.provider}: response held no audio")
        duration = max(1, round(frames * 1000 / rate))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(parsed.pcm[: frames * 2]))
        spans = spans_from_words(text, parsed.words)
        return SynthesizedAudio(
            data=buf.getvalue(),
            format=AUDIO_FORMAT,
            sample_rate=rate,
            duration_ms=duration,
            char_timings=char_timings_from_spans(len(text), spans, duration) if spans else [],
        )

    def _post(self, payload: dict[str, Any]) -> ParsedStream:
        key = self.cfg.api_key(self.provider)
        last = ""
        for attempt in range(MAX_RETRIES + 1):
            retry_after: float | None = None
            headers = {
                "X-Api-Key": key,
                "X-Api-Resource-Id": self.cfg.tts.model,
                "X-Api-Request-Id": str(uuid.uuid4()),
            }
            with self._limiter:
                try:
                    with self._http.stream("POST", PATH, json=payload, headers=headers) as resp:
                        if resp.status_code < 400:
                            parsed = parse_stream(resp.iter_lines())
                            if parsed.error is None and parsed.done:
                                return parsed
                            body: Any = parsed.error
                            status = resp.status_code
                            text = "" if body is not None else "stream ended early"
                        else:
                            text = resp.read().decode("utf-8", errors="replace")
                            status = resp.status_code
                            try:
                                body = json.loads(text)
                            except ValueError:
                                body = None
                            retry_after = parse_retry_after(resp.headers.get("retry-after"))
                except httpx.TransportError as e:
                    kind: FailureKind = "transient"
                    last = f"network error: {type(e).__name__}"
                else:
                    kind = _classify(status, body, text)
                    last = f"HTTP {status}: {_message(body, text)}"
            if kind == "quota":
                raise TTSQuotaExhausted(
                    f"{self.provider}: quota or balance exhausted ({last}); not retrying",
                    provider=self.provider,
                    reset_hint=_message(body) or None,
                )
            if kind == "auth":
                raise TTSAuthError(f"{self.provider}: not authorized ({last})")
            if kind == "fatal":
                raise TTSError(f"{self.provider}: request rejected ({last})")
            if attempt < MAX_RETRIES:
                self._sleep(retry_after if retry_after is not None else backoff_delay(attempt))
        raise TTSRateLimited(f"{self.provider}: still failing after {MAX_RETRIES} retries ({last})")


def _classify(status: int, body: Any, text: str) -> FailureKind:
    code = _code(body)
    if code in _AUTH_CODES:
        return "auth"
    if code in _FATAL_CODES:
        return "fatal"
    message = _message(body, text)
    if status < 400:
        if body is None:
            return "transient"  # cut off before the last event
        return "quota" if mentions_quota(message) else "fatal"  # an unknown error code
    return classify_failure(status, None, message)
