"""The Doubao speech adapter against recorded response shapes (docs/PROVIDERS.md §5), and the
matching of reported words back onto the input."""

from __future__ import annotations

import base64
import io
import json
import wave
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.tts import WordSpan, spans_from_words
from offscreen.config import AppConfig
from offscreen.providers.adapters.tts_cache import CachedTTS
from offscreen.providers.adapters.volc_tts import VolcTTS, parse_stream, speech_rate
from offscreen.providers.ports import (
    TTSAuthError,
    TTSError,
    TTSQuotaExhausted,
    TTSRateLimited,
)
from offscreen.services.tts import build_tts

RATE = 24000


def config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data_dir": str(tmp_path),
            "providers": {"volc": {"kind": "volc_tts", "api_key_env": "VOLC_TEST_KEY"}},
            "tts": {
                "provider": "volc",
                "model": "seed-tts-2.0",
                "default_voice": "zh_female_vv_uranus_bigtts",
                "sample_rate": RATE,
            },
        }
    )


def pcm(ms: int) -> bytes:
    return b"\x01\x00" * (RATE * ms // 1000)


def word(w: str, start: float, end: float) -> dict[str, Any]:
    return {"confidence": 0.9, "startTime": start, "endTime": end, "word": w}


def stream(text: str, audio: bytes, words: list[dict[str, Any]] | None = None) -> str:
    """The measured shape: audio chunks, the whole text without words, then each sentence with
    its words, then the end marker."""
    half = len(audio) // 2 // 2 * 2
    events: list[dict[str, Any]] = [
        {"code": 0, "message": "", "data": base64.b64encode(audio[:half]).decode()},
        {"code": 0, "message": "", "data": base64.b64encode(audio[half:]).decode()},
        {"code": 0, "message": "", "sentence": {"phonemes": [], "text": text, "words": []}},
    ]
    if words:
        events.append(
            {"code": 0, "message": "", "sentence": {"phonemes": [], "text": text, "words": words}}
        )
    events.append({"code": 20000000, "message": "OK"})
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n"


class Api:
    def __init__(self, *replies: httpx.Response) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.replies.pop(0)


def adapter(tmp_path: Path, api: Api, monkeypatch: pytest.MonkeyPatch) -> VolcTTS:
    monkeypatch.setenv("VOLC_TEST_KEY", "k-123")
    return VolcTTS(config(tmp_path), transport=httpx.MockTransport(api), sleep=lambda _: None)


def test_synthesize_sends_the_measured_request_and_builds_a_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "我买了一台 iPhone，好。"
    words = [
        word("我", 0.295, 0.425),
        word("买", 0.425, 0.615),
        word("了", 0.615, 0.695),
        word("一", 0.695, 0.785),
        word("台", 0.785, 0.955),
        word("iPhone，", 0.955, 1.415),
        word("好。", 1.655, 1.925),
    ]
    api = Api(httpx.Response(200, text=stream(text, pcm(2000), words)))
    tts = adapter(tmp_path, api, monkeypatch)
    out = tts.synthesize(text, voice_id="zh_female_vv_uranus_bigtts", speed=1.2)

    (req,) = api.requests
    assert req.url == "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    assert req.headers["X-Api-Key"] == "k-123"
    assert req.headers["X-Api-Resource-Id"] == "seed-tts-2.0"
    assert req.headers["X-Api-Request-Id"]
    body = json.loads(req.content)
    assert body["req_params"] == {
        "text": text,
        "speaker": "zh_female_vv_uranus_bigtts",
        "audio_params": {
            "format": "pcm",
            "sample_rate": RATE,
            "speech_rate": 20,
            "enable_subtitle": True,
        },
    }
    assert (out.format, out.sample_rate, out.duration_ms) == ("wav", RATE, 2000)
    with wave.open(io.BytesIO(out.data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
        assert w.getnframes() == RATE * 2
    assert tts.id == f"volc:seed-tts-2.0:wav@{RATE}"
    t = out.char_timings
    assert len(t) == len(text)
    assert t[0] == (295, 425)
    assert t[5] == (955, 955)  # the space before iPhone is not spoken
    assert t[6][0] == 955 and t[12][1] == 1415  # "iPhone，" shared over its 7 characters
    assert t[-1][1] == 1925


def test_no_words_means_no_timings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = Api(httpx.Response(200, text=stream("你好", pcm(500))))
    out = adapter(tmp_path, api, monkeypatch).synthesize("你好", voice_id="v")
    assert out.char_timings == [] and out.duration_ms == 500


@pytest.mark.parametrize(
    ("speed", "rate"), [(0.5, -50), (0.9, -10), (1.0, 0), (1.5, 50), (2.0, 100)]
)
def test_speed_becomes_speech_rate(speed: float, rate: int) -> None:
    assert speech_rate(speed) == rate


def test_errors_are_classified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def err(status: int, code: int, message: str, header: bool = True) -> httpx.Response:
        payload = (
            {"header": {"reqid": "r", "code": code, "message": message}}
            if header
            else {"reqid": "", "code": code, "message": message}
        )
        return httpx.Response(status, json=payload)

    cases: list[tuple[httpx.Response, type[Exception], str]] = [
        (err(401, 45000010, "Invalid X-Api-Key"), TTSAuthError, "Invalid X-Api-Key"),
        (err(403, 45000030, "requested resource not granted"), TTSAuthError, "not granted"),
        (
            err(200, 55000000, "resource ID is mismatched with speaker", header=False),
            TTSError,
            "mismatched",
        ),
        (err(200, 45002001, "No readable text!", header=False), TTSError, "readable"),
        (err(200, 45009999, "quota exceeded", header=False), TTSQuotaExhausted, "quota"),
    ]
    for reply, kind, words in cases:
        api = Api(reply)
        with pytest.raises(kind, match=words):
            adapter(tmp_path, api, monkeypatch).synthesize("你好", voice_id="v")
        assert len(api.requests) == 1  # not retried
    with pytest.raises(TTSAuthError):  # never TTSRateLimited, which would be retried
        adapter(tmp_path, Api(err(401, 45000010, "x")), monkeypatch).synthesize(
            "你好", voice_id="v"
        )


def test_transient_failures_are_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cut_off = '{"code":0,"message":"","data":"AAAA"}\n'  # no end marker
    api = Api(
        httpx.Response(429, json={"header": {"code": 45000292, "message": "too many requests"}}),
        httpx.Response(503, text="busy"),
        httpx.Response(200, text=cut_off),
        httpx.Response(200, text=stream("你好", pcm(400))),
    )
    out = adapter(tmp_path, api, monkeypatch).synthesize("你好", voice_id="v")
    assert out.duration_ms == 400 and len(api.requests) == 4
    always = Api(*[httpx.Response(503, text="busy") for _ in range(4)])
    with pytest.raises(TTSRateLimited):
        adapter(tmp_path, always, monkeypatch).synthesize("你好", voice_id="v")


def test_bad_input_and_empty_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = Api(httpx.Response(200, text='{"code":20000000,"message":"OK"}\n'))
    tts = adapter(tmp_path, api, monkeypatch)
    for text, speed in (("  ", 1.0), ("字" * 1001, 1.0), ("你好", 2.5)):
        with pytest.raises(ValueError):
            tts.synthesize(text, voice_id="v", speed=speed)
    with pytest.raises(TTSError, match="no audio"):
        tts.synthesize("你好", voice_id="v")
    with pytest.raises(TTSError, match="unparseable"):
        parse_stream(["not json"])


def test_the_configured_volc_provider_is_built(tmp_path: Path) -> None:
    tts = build_tts(config(tmp_path))
    assert isinstance(tts, CachedTTS) and isinstance(tts.inner, VolcTTS)


# ---- words back onto the text ---------------------------------------------------------------
def test_words_are_found_in_order_with_their_punctuation() -> None:
    text = "我买了一台 iPhone，Hello world！他说：“走吧。”"
    words = [("我", 0, 1), ("台", 1, 2), ("iPhone，", 2, 3), ("Hello", 3, 4), ("world！", 4, 5)]
    words += [("他", 5, 6), ("说：“", 6, 7), ("走", 7, 8), ("吧。”", 8, 9)]
    spans = spans_from_words(text, words)
    assert [text[s.char_lo : s.char_hi] for s in spans] == [w for w, _, _ in words]
    assert spans[2] == WordSpan(2, 3, 6, 13)


def test_a_rewritten_word_is_skipped_without_latching_onto_a_later_one() -> None:
    text = "他在1990年出生，后来他走了。"
    words = [("他", 0, 1), ("在", 1, 2), ("一九九零", 2, 3), ("年", 3, 4), ("出", 4, 5)]
    spans = spans_from_words(text, words)
    assert [text[s.char_lo : s.char_hi] for s in spans] == ["他", "在", "年", "出"]
    far = spans_from_words("你好，今天天气不错，我们去公园散步吧，他", [("他", 0, 1)])
    assert far == []  # too far from where the previous word ended


def test_a_word_whose_punctuation_differs_is_matched_by_its_core() -> None:
    spans = spans_from_words("好的!", [("好的！", 0, 1)])
    assert [(s.char_lo, s.char_hi) for s in spans] == [(0, 2)]


@given(st.lists(st.sampled_from(["他", "走", "了", "iPhone", "，", "。", " "]), max_size=20))
def test_reported_words_of_the_text_itself_are_all_found(pieces: list[str]) -> None:
    text = "".join(pieces)
    words = [(p, float(i), float(i + 1)) for i, p in enumerate(pieces) if p.strip()]
    spans = spans_from_words(text, words)
    assert len(spans) == len(words)
    assert all(a.char_hi <= b.char_lo for a, b in pairwise(spans))
