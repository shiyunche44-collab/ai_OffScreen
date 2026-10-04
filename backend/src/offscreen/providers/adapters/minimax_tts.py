"""MiniMax speech synthesis `POST {base_url}/t2a_v2` (docs/PROVIDERS.md §2.5).

Always streams: only the streaming response carries per-word timestamps inline (the
non-streaming one hands them out as a file URL on a domain cloud networks block). In the
SSE stream the `status: 2` event repeats the complete audio (its size equals
`extra_info.audio_size`) and the subtitles; the earlier `status: 1` events hold partial
audio that does not concatenate to the same bytes, so the final event is preferred."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import httpx

from offscreen.algo.tts import WordSpan, char_timings_from_spans
from offscreen.config import AppConfig
from offscreen.providers.adapters.http_util import (
    MAX_RETRIES,
    FailureKind,
    backoff_delay,
    classify_failure,
    error_text,
    parse_retry_after,
)
from offscreen.providers.ports import (
    SynthesizedAudio,
    TTSAuthError,
    TTSError,
    TTSQuotaExhausted,
    TTSRateLimited,
)

TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=60.0)
AUDIO_FORMAT = "mp3"
SPEED_RANGE = (0.5, 2.0)
MAX_TEXT_CHARS = 10_000


class MalformedStream(TTSError):
    """The response did not have the documented shape."""


@dataclass
class ParsedStream:
    final_audio: bytes | None = None
    chunks: list[bytes] = field(default_factory=list)
    spans: list[WordSpan] = field(default_factory=list)
    extra_info: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    """The first event whose `base_resp.status_code` was non-zero, as parsed JSON."""


def parse_t2a_stream(lines: Iterable[str]) -> ParsedStream:
    """Fold the SSE lines of one response into audio, word spans and `extra_info`."""
    out = ParsedStream()
    inline_spans: list[WordSpan] = []
    final_spans: list[WordSpan] | None = None
    for raw in lines:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:])
        except ValueError as e:
            raise MalformedStream(f"unparseable SSE event: {line[:200]}") from e
        base = event.get("base_resp") or {}
        if base.get("status_code"):
            out.error = event
            return out
        data = event.get("data") or {}
        audio_hex = data.get("audio") or ""
        try:
            audio = bytes.fromhex(audio_hex)
        except ValueError as e:
            raise MalformedStream("audio is not valid hex") from e
        if data.get("status") == 2:
            out.final_audio = audio or None
            if "subtitles" in data:
                final_spans = _spans(data["subtitles"])
            out.extra_info = event.get("extra_info") or {}
        else:
            if audio:
                out.chunks.append(audio)
            if "subtitle" in data:
                inline_spans.extend(_spans([data["subtitle"]]))
    out.spans = final_spans if final_spans is not None else inline_spans
    return out


def _spans(subtitles: list[dict[str, Any]]) -> list[WordSpan]:
    return [
        WordSpan(
            start_ms=float(w["time_begin"]),
            end_ms=float(w["time_end"]),
            char_lo=int(w["word_begin"]),
            char_hi=int(w["word_end"]),
        )
        for sub in subtitles
        for w in sub.get("timestamped_words") or []
    ]


class MiniMaxTTS:
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
        if pcfg is None or pcfg.kind != "openai_compat" or not pcfg.base_url:
            raise ValueError(f"tts.provider {self.provider!r} is not a MiniMax-style provider")
        self._limiter = threading.BoundedSemaphore(pcfg.max_concurrency)
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=pcfg.base_url.rstrip("/"), timeout=TIMEOUT, transport=transport
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
            "model": t.model,
            "text": text,
            "stream": True,
            "voice_setting": {"voice_id": voice_id, "speed": speed, "vol": 1, "pitch": 0},
            "audio_setting": {"sample_rate": t.sample_rate, "format": AUDIO_FORMAT, "channel": 1},
            "subtitle_enable": True,
            "subtitle_type": "word",
        }
        parsed = self._post(payload)
        return self._result(text, parsed)

    def _result(self, text: str, parsed: ParsedStream) -> SynthesizedAudio:
        audio = parsed.final_audio or b"".join(parsed.chunks)
        if not audio:
            raise TTSError(f"{self.provider}: response held no audio")
        info = parsed.extra_info
        duration = int(info.get("audio_length") or 0)
        if duration <= 0:
            raise MalformedStream(f"{self.provider}: response has no extra_info.audio_length")
        return SynthesizedAudio(
            data=audio,
            format=str(info.get("audio_format") or AUDIO_FORMAT),
            sample_rate=int(info.get("audio_sample_rate") or self.cfg.tts.sample_rate),
            duration_ms=duration,
            char_timings=char_timings_from_spans(len(text), parsed.spans, duration)
            if parsed.spans
            else [],
            billed_chars=int(info["usage_characters"]) if "usage_characters" in info else None,
        )

    def _post(self, payload: dict[str, Any]) -> ParsedStream:
        headers = {"Authorization": f"Bearer {self.cfg.api_key(self.provider)}"}
        last = ""
        for attempt in range(MAX_RETRIES + 1):
            retry_after: float | None = None
            body: Any = None
            forced: FailureKind | None = None
            with self._limiter:
                try:
                    with self._http.stream(
                        "POST", "/t2a_v2", json=payload, headers=headers
                    ) as resp:
                        if resp.status_code < 400:
                            parsed = parse_t2a_stream(resp.iter_lines())
                            if parsed.error is not None:
                                body, status, text = parsed.error, 200, ""
                            elif not parsed.extra_info:  # cut off before the final event
                                forced, status, text = "transient", 200, "stream ended early"
                            else:
                                return parsed
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
                    kind = forced or classify_failure(status, body, text)
                    last = f"HTTP {status}: {error_text(body, text)}"
            if kind == "quota":
                raise TTSQuotaExhausted(
                    f"{self.provider}: quota or balance exhausted ({last}); not retrying",
                    provider=self.provider,
                    reset_hint=error_text(body, "") or None,
                )
            if kind == "auth":
                raise TTSAuthError(f"{self.provider}: authentication failed ({last})")
            if kind == "fatal":
                raise TTSError(f"{self.provider}: request rejected ({last})")
            if attempt < MAX_RETRIES:
                self._sleep(retry_after if retry_after is not None else backoff_delay(attempt))
        raise TTSRateLimited(f"{self.provider}: still failing after {MAX_RETRIES} retries ({last})")
