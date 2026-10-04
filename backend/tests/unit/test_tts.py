from __future__ import annotations

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

from offscreen.algo.tts import WordSpan, char_timings_from_spans, tts_cache_key
from offscreen.config import AppConfig
from offscreen.providers.adapters.fake import FakeTTS
from offscreen.providers.adapters.minimax_tts import (
    MalformedStream,
    MiniMaxTTS,
    parse_t2a_stream,
)
from offscreen.providers.adapters.tts_cache import CachedTTS
from offscreen.providers.ports import (
    TTSAuthError,
    TTSError,
    TTSQuotaExhausted,
    TTSRateLimited,
)
from offscreen.services.tts import build_tts

KEY = "sk-test-SECRET-123"
TEXT = "这个女孩，走遍了世界。Hello 2024!"


# ---- algo ------------------------------------------------------------------------------
def test_cache_key_depends_on_everything_that_changes_audio() -> None:
    base = tts_cache_key("你好", "v1", 1.0, "e1")
    assert base == tts_cache_key("你好", "v1", 1.0, "e1")
    assert base.startswith("sha256:")
    assert (
        len(
            {
                base,
                tts_cache_key("你好!", "v1", 1.0, "e1"),
                tts_cache_key("你好", "v2", 1.0, "e1"),
                tts_cache_key("你好", "v1", 1.1, "e1"),
                tts_cache_key("你好", "v1", 1.0, "e2"),
            }
        )
        == 5
    )


def real_spans() -> list[WordSpan]:
    """The shape MiniMax returned for TEXT (docs/PROVIDERS.md §2.5): per-character CJK, sub-word
    English pieces, a number repeated six times with the same offsets, no entry for the space."""
    cjk = [
        (42.7, 170.7),
        (170.7, 298.7),
        (298.7, 469.3),
        (469.3, 597.3),
        (597.3, 640.0),
        (640.0, 938.7),
        (938.7, 1536.0),
        (1536.0, 1706.7),
        (1706.7, 1834.7),
        (1834.7, 2005.3),
        (2005.3, 2133.3),
    ]
    spans = [WordSpan(a, b, i, i + 1) for i, (a, b) in enumerate(cjk)]
    spans += [WordSpan(2133.3, 2389.3 + 128, 11, 13), WordSpan(2730.7, 3541.3, 13, 16)]
    spans += [
        WordSpan(a, b, 17, 21)
        for a, b in [
            (3541.3, 3754.7),
            (3754.7, 3925.3),
            (3925.3, 4096),
            (4096, 4224),
            (4224, 4352),
            (4352, 4480),
        ]
    ]
    spans.append(WordSpan(4480, 4736, 21, 22))
    return spans


def test_char_timings_from_real_shape() -> None:
    t = char_timings_from_spans(len(TEXT), real_spans(), 4969)
    assert len(t) == len(TEXT) == 22
    assert t[0] == (43, 171) and t[10] == (2005, 2133)
    # "He" (offsets 11..13) is split evenly over two characters.
    assert t[11][0] == 2133 and t[11][1] == t[12][0] and t[12][1] == 2517
    # The space at index 16 has no span: empty interval where "llo" ended.
    assert t[16] == (t[15][1], t[15][1])
    # "2024" (17..21): six repeated entries merge into 3541..4480, split over four digits.
    assert t[17][0] == 3541 and t[20][1] == 4480
    assert t[21] == (4480, 4736)
    assert all(a <= b for a, b in t)
    assert all(x[1] <= y[0] for x, y in pairwise(t))


def test_char_timings_clamps_to_duration_and_ignores_bad_spans() -> None:
    spans = [
        WordSpan(0, 500, 0, 1),
        WordSpan(500, 900, 1, 2),
        WordSpan(0, 1, 5, 9),
        WordSpan(0, 1, 1, 1),
    ]
    assert char_timings_from_spans(2, spans, 700) == [(0, 500), (500, 700)]
    assert char_timings_from_spans(0, spans, 700) == []


def test_overlapping_spans_first_claim_wins() -> None:
    spans = [WordSpan(0, 400, 0, 2), WordSpan(100, 200, 1, 2)]
    assert char_timings_from_spans(2, spans, 1000) == [(0, 200), (200, 400)]


@st.composite
def spans_and_len(draw: st.DrawFn) -> tuple[int, list[WordSpan], int]:
    n = draw(st.integers(1, 40))
    duration = draw(st.integers(1, 60_000))
    spans = []
    for _ in range(draw(st.integers(0, 30))):
        lo = draw(st.integers(-3, n + 3))
        hi = draw(st.integers(lo - 1, n + 5))
        a = draw(st.floats(-100, duration + 500, allow_nan=False))
        b = draw(st.floats(a, duration + 1000, allow_nan=False))
        spans.append(WordSpan(a, b, lo, hi))
    return n, spans, duration


@given(spans_and_len())
def test_char_timings_properties(case: tuple[int, list[WordSpan], int]) -> None:
    n, spans, duration = case
    t = char_timings_from_spans(n, spans, duration)
    assert len(t) == n
    assert all(0 <= a <= b <= duration for a, b in t)
    assert all(x[1] <= y[0] for x, y in pairwise(t))


# ---- MiniMax adapter -------------------------------------------------------------------
AUDIO = bytes(range(200))


def event(data: dict[str, Any], **extra: Any) -> str:
    return "data: " + json.dumps(
        {"data": data, "trace_id": "t", "base_resp": {"status_code": 0, "status_msg": ""}, **extra}
    )


def words(spans: list[WordSpan]) -> list[dict[str, Any]]:
    return [
        {
            "word": "?",
            "time_begin": s.start_ms,
            "time_end": s.end_ms,
            "word_begin": s.char_lo,
            "word_end": s.char_hi,
        }
        for s in spans
    ]


def good_stream(text: str = TEXT, spans: list[WordSpan] | None = None) -> bytes:
    sub = {
        "text": text,
        "time_begin": 0,
        "time_end": 4869.0,
        "text_begin": 0,
        "text_end": len(text),
        "timestamped_words": words(spans if spans is not None else real_spans()),
    }
    lines = [
        event({"audio": AUDIO[:50].hex(), "status": 1, "ced": ""}),
        event({"audio": AUDIO[50:120].hex(), "status": 1, "ced": "", "subtitle": sub}),
        event({"audio": "", "status": 1, "ced": ""}),
        event(
            {"audio": AUDIO.hex(), "status": 2, "ced": "", "subtitles": [sub]},
            extra_info={
                "audio_length": 4969,
                "audio_sample_rate": 32000,
                "audio_size": len(AUDIO),
                "audio_format": "mp3",
                "usage_characters": 31,
            },
        ),
    ]
    return ("\n\n".join(lines) + "\n\n").encode()


def make_cfg(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data_dir": str(tmp_path),
            "providers": {
                "p": {
                    "kind": "openai_compat",
                    "base_url": "https://tts.example/v1",
                    "api_key_env": "TEST_TTS_KEY",
                    "max_concurrency": 2,
                },
            },
            "asr": {"provider": "faster_whisper"},
            "tts": {"provider": "p", "model": "speech-x", "sample_rate": 32000},
        }
    )


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_TTS_KEY", KEY)


def tts_with(tmp_path: Path, handler: Any, sleeps: list[float] | None = None) -> MiniMaxTTS:
    return MiniMaxTTS(
        make_cfg(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else (lambda _s: None)),
    )


def test_synthesize_parses_stream_and_sends_documented_request(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200, content=good_stream(), headers={"content-type": "text/event-stream"}
        )

    out = tts_with(tmp_path, handler).synthesize(TEXT, voice_id="male-qn-qingse", speed=1.2)

    assert out.data == AUDIO  # the final event's complete audio, not the partial chunks
    assert (out.format, out.sample_rate, out.duration_ms, out.billed_chars) == (
        "mp3",
        32000,
        4969,
        31,
    )
    assert len(out.char_timings) == len(TEXT) and out.char_timings[0] == (43, 171)

    req = seen[0]
    assert str(req.url) == "https://tts.example/v1/t2a_v2"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(req.content)
    assert body["model"] == "speech-x" and body["stream"] is True and body["text"] == TEXT
    assert body["voice_setting"] == {
        "voice_id": "male-qn-qingse",
        "speed": 1.2,
        "vol": 1,
        "pitch": 0,
    }
    assert body["audio_setting"] == {"sample_rate": 32000, "format": "mp3", "channel": 1}
    assert body["subtitle_enable"] is True and body["subtitle_type"] == "word"


def test_no_subtitles_gives_empty_timings_not_partial(tmp_path: Path) -> None:
    out = tts_with(
        tmp_path, lambda r: httpx.Response(200, content=good_stream(spans=[]))
    ).synthesize(TEXT, voice_id="v")
    assert out.char_timings == []


def test_falls_back_to_chunks_when_final_event_has_no_audio(tmp_path: Path) -> None:
    sub_less = [
        event({"audio": AUDIO[:60].hex(), "status": 1, "ced": ""}),
        event({"audio": AUDIO[60:].hex(), "status": 1, "ced": ""}),
        event({"audio": "", "status": 2, "ced": ""}, extra_info={"audio_length": 1000}),
    ]
    out = tts_with(
        tmp_path, lambda r: httpx.Response(200, content="\n\n".join(sub_less).encode())
    ).synthesize("你好", voice_id="v")
    assert out.data == AUDIO and out.duration_ms == 1000 and out.sample_rate == 32000


def test_parse_errors() -> None:
    with pytest.raises(MalformedStream):
        parse_t2a_stream(["data: {not json"])
    with pytest.raises(MalformedStream):
        parse_t2a_stream([event({"audio": "zz", "status": 1})])
    assert parse_t2a_stream(["", ": comment", "event: x"]).final_audio is None


def failing(status: int, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(status, json=body)


def test_rate_limit_is_retried_then_succeeds(tmp_path: Path) -> None:
    calls = []
    sleeps: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(
                429,
                json={"base_resp": {"status_code": 1002, "status_msg": "rate limit"}},
                headers={"retry-after": "3"},
            )
        return httpx.Response(200, content=good_stream())

    out = tts_with(tmp_path, handler, sleeps).synthesize(TEXT, voice_id="v")
    assert out.duration_ms == 4969 and len(calls) == 3 and sleeps == [3.0, 3.0]


def test_error_inside_http_200_stream_is_classified(tmp_path: Path) -> None:
    quota = "data: " + json.dumps(
        {
            "data": {},
            "base_resp": {"status_code": 2056, "status_msg": "usage limit exceeded (window)"},
        }
    )
    with pytest.raises(TTSQuotaExhausted) as ei:
        tts_with(tmp_path, lambda r: httpx.Response(200, content=quota.encode())).synthesize(
            "你好", voice_id="v"
        )
    assert ei.value.provider == "p" and "usage limit" in (ei.value.reset_hint or "")


def test_auth_and_fatal_fail_fast(tmp_path: Path) -> None:
    n = []

    def auth(req: httpx.Request) -> httpx.Response:
        n.append(1)
        return failing(401, {"base_resp": {"status_code": 1004, "status_msg": "invalid api key"}})

    with pytest.raises(TTSAuthError):
        tts_with(tmp_path, auth).synthesize("你好", voice_id="v")
    assert len(n) == 1

    with pytest.raises(TTSError, match="rejected") as ei:
        tts_with(
            tmp_path,
            lambda r: failing(400, {"base_resp": {"status_code": 2013, "status_msg": "bad voice"}}),
        ).synthesize("你好", voice_id="nope")
    assert not isinstance(ei.value, (TTSAuthError, TTSQuotaExhausted, TTSRateLimited))


def test_gives_up_after_retries_and_never_leaks_the_key(tmp_path: Path) -> None:
    n = []

    def handler(req: httpx.Request) -> httpx.Response:
        n.append(1)
        return httpx.Response(503, text="overloaded")

    with pytest.raises(TTSRateLimited) as ei:
        tts_with(tmp_path, handler).synthesize("你好", voice_id="v")
    assert len(n) == 4 and KEY not in str(ei.value)


def test_network_error_and_truncated_stream_are_retried(tmp_path: Path) -> None:
    n = []
    truncated = event({"audio": AUDIO[:10].hex(), "status": 1, "ced": ""}).encode()

    def handler(req: httpx.Request) -> httpx.Response:
        n.append(1)
        if len(n) == 1:
            raise httpx.ConnectError("boom")
        if len(n) == 2:
            return httpx.Response(200, content=truncated)
        return httpx.Response(200, content=good_stream())

    assert tts_with(tmp_path, handler).synthesize(TEXT, voice_id="v").duration_ms == 4969
    assert len(n) == 3


def test_input_validation_and_config(tmp_path: Path) -> None:
    t = tts_with(tmp_path, lambda r: httpx.Response(500))
    for bad in (
        dict(text=" ", speed=1.0),
        dict(text="x", speed=0.4),
        dict(text="x", speed=2.1),
        dict(text="字" * 10_001, speed=1.0),
    ):
        with pytest.raises(ValueError):
            t.synthesize(bad["text"], voice_id="v", speed=bad["speed"])  # type: ignore[arg-type]
    assert t.id == "p:speech-x:mp3@32000"
    cfg = AppConfig.model_validate(
        {"asr": {"provider": "faster_whisper"}, "tts": {"provider": "edge_tts"}}
    )
    with pytest.raises(ValueError, match="MiniMax-style"):
        MiniMaxTTS(cfg)


def test_build_tts_wires_cache_dir(tmp_path: Path) -> None:
    tts = build_tts(make_cfg(tmp_path))
    assert tts.dir == tmp_path / "tts_cache" and tts.id == "p:speech-x:mp3@32000"


# ---- cache + fake ------------------------------------------------------------------------
def test_fake_tts_wav_and_timings() -> None:
    out = FakeTTS(chars_per_s=5.0, sample_rate=8000).synthesize("你好 世界!", voice_id="v")
    # 5 spoken characters (space excluded) at 5 chars/s -> 1 s.
    assert out.duration_ms == 1000 and out.format == "wav"
    with wave.open(io.BytesIO(out.data)) as w:
        assert w.getframerate() == 8000 and w.getnframes() == 8000 and w.getnchannels() == 1
    t = out.char_timings
    assert len(t) == 6 and t[0] == (0, 200) and t[2] == (400, 400) and t[5] == (800, 1000)
    faster = FakeTTS(chars_per_s=5.0).synthesize("你好 世界!", voice_id="v", speed=2.0)
    assert faster.duration_ms == 500


def test_cached_tts_hits_and_keys(tmp_path: Path) -> None:
    fake = FakeTTS()
    cached = CachedTTS(fake, tmp_path / "c")
    a = cached.synthesize("你好世界", voice_id="v1")
    b = cached.synthesize("你好世界", voice_id="v1")
    assert len(fake.calls) == 1 and (cached.hits, cached.misses) == (1, 1)
    assert (b.data, b.duration_ms, b.char_timings) == (a.data, a.duration_ms, a.char_timings)
    assert a.billed_chars == 4 and b.billed_chars == 0  # a hit is free

    cached.synthesize("你好世界", voice_id="v2")
    cached.synthesize("你好世界", voice_id="v1", speed=1.1)
    cached.synthesize("你好世界!", voice_id="v1")
    assert len(fake.calls) == 4

    again = CachedTTS(fake, tmp_path / "c")  # a new process sees the same files
    again.synthesize("你好世界", voice_id="v1")
    assert again.hits == 1 and len(fake.calls) == 4

    other_engine = CachedTTS(FakeTTS(chars_per_s=3.0), tmp_path / "c")
    other_engine.synthesize("你好世界", voice_id="v1")
    assert other_engine.misses == 1


def test_cached_tts_recovers_from_damaged_entry(tmp_path: Path) -> None:
    fake = FakeTTS()
    cached = CachedTTS(fake, tmp_path / "c")
    cached.synthesize("你好", voice_id="v")
    for meta in (tmp_path / "c").glob("*.json"):
        meta.write_text("{broken")
    cached.synthesize("你好", voice_id="v")
    assert len(fake.calls) == 2
    cached.synthesize("你好", voice_id="v")
    assert len(fake.calls) == 2


def test_cached_tts_does_not_cache_failures(tmp_path: Path) -> None:
    fake = FakeTTS(fail_with=TTSError("boom"))
    cached = CachedTTS(fake, tmp_path / "c")
    with pytest.raises(TTSError):
        cached.synthesize("你好", voice_id="v")
    fake.fail_with = None
    assert cached.synthesize("你好", voice_id="v").duration_ms > 0
    assert not list((tmp_path / "c").glob(".*"))
