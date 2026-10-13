"""The CosyVoice adapter, its cache key, its wiring, and the sidecar's HTTP contract (the real
sidecar script, run with its fake engine)."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import wave
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from offscreen.config import AppConfig
from offscreen.domain.voice import Voice, VoiceLibrary
from offscreen.providers.adapters.cosyvoice_tts import CosyVoiceTTS, VoicePrompt
from offscreen.providers.adapters.tts_cache import CachedTTS
from offscreen.providers.ports import TTSError
from offscreen.services.tts import build_tts
from offscreen.store.voices import VoiceStore

SIDECAR = Path(__file__).resolve().parents[3] / "sidecars" / "cosyvoice" / "server.py"
MODEL = "CosyVoice2-0.5B"


def wav(seconds: float, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * round(seconds * rate))
    return buf.getvalue()


def config(tmp_path: Path, url: str = "http://sidecar", model: str = MODEL) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data_dir": str(tmp_path / "data"),
            "tts": {"provider": "cosyvoice", "model": model, "cosyvoice": {"url": url}},
        }
    )


class Sidecar:
    """A stand-in sidecar: answers /health and records /synthesize requests."""

    def __init__(self, *, model: str = MODEL, api_version: int = 1) -> None:
        self.model = model
        self.api_version = api_version
        self.requests: list[dict[str, object]] = []
        self.health_calls = 0
        self.reply: httpx.Response | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            self.health_calls += 1
            return httpx.Response(
                200,
                json={"status": "ok", "api_version": self.api_version, "model": self.model},
            )
        self.requests.append(json.loads(request.content))
        return self.reply or httpx.Response(200, content=wav(1.5))


def adapter(
    tmp_path: Path, sidecar: Sidecar, prompts: dict[str, VoicePrompt] | None = None
) -> CosyVoiceTTS:
    table = prompts or {}
    return CosyVoiceTTS(config(tmp_path), prompts=table.get, transport=httpx.MockTransport(sidecar))


def test_a_built_in_speaker_is_named_and_the_wav_measured(tmp_path: Path) -> None:
    side = Sidecar()
    tts = adapter(tmp_path, side)
    audio = tts.synthesize("你好", voice_id="中文女", speed=1.1)
    assert side.requests == [{"text": "你好", "speed": 1.1, "speaker": "中文女"}]
    assert (audio.format, audio.sample_rate, audio.duration_ms) == ("wav", 24000, 1500)
    assert audio.char_timings == [] and audio.data.startswith(b"RIFF")
    assert tts.id == f"cosyvoice:{MODEL}:wav"
    tts.synthesize("再见", voice_id="中文女")
    assert side.health_calls == 1  # the model is checked once


def test_a_cloned_voice_sends_its_recording_and_transcript(tmp_path: Path) -> None:
    ref = tmp_path / "me.wav"
    ref.write_bytes(wav(3.0, 16000))
    side = Sidecar()
    tts = adapter(
        tmp_path,
        side,
        {"me": VoicePrompt(ref, "大家好"), "quiet": VoicePrompt(ref, None)},
    )
    tts.synthesize("你好", voice_id="me")
    tts.synthesize("你好", voice_id="quiet")
    first, second = side.requests
    assert base64.b64decode(str(first["prompt_audio"])) == ref.read_bytes()
    assert first["prompt_text"] == "大家好" and "speaker" not in first
    assert "prompt_text" not in second  # no transcript: the sidecar clones cross-lingually


def test_a_sidecar_serving_another_model_or_api_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TTSError, match=r"tts\.model"):
        adapter(tmp_path, Sidecar(model="CosyVoice-300M-SFT")).synthesize("你好", voice_id="x")
    with pytest.raises(TTSError, match="API 2"):
        adapter(tmp_path, Sidecar(api_version=2)).synthesize("你好", voice_id="x")


def test_failures_say_what_went_wrong(tmp_path: Path) -> None:
    side = Sidecar()
    side.reply = httpx.Response(400, json={"error": "unknown speaker 'x'"})
    with pytest.raises(TTSError, match="HTTP 400: unknown speaker"):
        adapter(tmp_path, side).synthesize("你好", voice_id="x")
    side.reply = httpx.Response(200, content=b"not a wav")
    with pytest.raises(TTSError, match="WAV"):
        adapter(tmp_path, side).synthesize("你好", voice_id="x")
    side.reply = httpx.Response(200, content=wav(0))
    with pytest.raises(TTSError, match="empty"):
        adapter(tmp_path, side).synthesize("你好", voice_id="x")
    with pytest.raises(TTSError, match="reference recording"):
        adapter(tmp_path, Sidecar(), {"me": VoicePrompt(tmp_path / "gone.wav")}).synthesize(
            "你好", voice_id="me"
        )

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    down = CosyVoiceTTS(
        config(tmp_path), prompts=lambda _: None, transport=httpx.MockTransport(refuse)
    )
    with pytest.raises(TTSError, match=r"not reachable.*README"):
        down.synthesize("你好", voice_id="x")


def test_bad_input_is_refused_before_any_request(tmp_path: Path) -> None:
    side = Sidecar()
    tts = adapter(tmp_path, side)
    for text, speed in (("  ", 1.0), ("字" * 2001, 1.0), ("你好", 2.5)):
        with pytest.raises(ValueError):
            tts.synthesize(text, voice_id="x", speed=speed)
    assert side.health_calls == 0 and side.requests == []


def test_a_replaced_recording_is_spoken_again_not_taken_from_the_cache(tmp_path: Path) -> None:
    ref = tmp_path / "me.wav"
    ref.write_bytes(wav(3.0))
    prompts = {"me": VoicePrompt(ref, "大家好")}
    side = Sidecar()
    cosy = adapter(tmp_path, side, prompts)
    cached = CachedTTS(cosy, tmp_path / "cache", voice_key=cosy.voice_key)
    assert cosy.voice_key("builtin") == ""
    cached.synthesize("你好", voice_id="me")
    cached.synthesize("你好", voice_id="me")
    assert len(side.requests) == 1
    ref.write_bytes(wav(4.0))
    cached.synthesize("你好", voice_id="me")
    prompts["me"] = VoicePrompt(ref, "大家晚上好")
    cached.synthesize("你好", voice_id="me")
    assert len(side.requests) == 3


def test_the_configured_engine_clones_voices_from_the_library(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    store = VoiceStore(cfg.data_dir)
    store.dir.mkdir(parents=True)
    (store.dir / "me.wav").write_bytes(wav(3.0))
    store.write(
        VoiceLibrary(
            voices=[
                Voice(
                    id="me",
                    name="我",
                    provider="cosyvoice",
                    reference_audio="me.wav",
                    reference_text="大家好",
                ),
                Voice(id="plain", name="plain", provider="cosyvoice"),
            ]
        )
    )
    tts = build_tts(cfg)
    assert isinstance(tts.inner, CosyVoiceTTS)
    lookup = tts.inner.prompts
    assert lookup("me") == VoicePrompt(store.dir / "me.wav", "大家好")
    assert lookup("plain") is None and lookup("unknown") is None


# ---- the sidecar itself ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def sidecar_url() -> Iterator[str]:
    proc = subprocess.Popen(
        [sys.executable, "-I", str(SIDECAR), "--engine", "fake", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert line.startswith("listening on "), line
        yield "http://" + line.removeprefix("listening on ").strip()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_the_adapter_speaks_through_the_real_sidecar(tmp_path: Path, sidecar_url: str) -> None:
    ref = tmp_path / "me.wav"
    ref.write_bytes(wav(3.0, 16000))
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    prompts = {"me": VoicePrompt(ref, "大家好"), "bad": VoicePrompt(bad)}
    tts = CosyVoiceTTS(config(tmp_path, sidecar_url, model="fake"), prompts=prompts.get)

    built_in = tts.synthesize("十个字的一句话而已。", voice_id="fake")  # 10 chars at 5/s
    assert (built_in.sample_rate, built_in.duration_ms) == (16000, 2000)
    assert tts.synthesize("十个字的一句话而已。", voice_id="me", speed=2.0).duration_ms == 1000
    with pytest.raises(TTSError, match="HTTP 400: unknown speaker"):
        tts.synthesize("你好", voice_id="nobody")
    with pytest.raises(TTSError, match="HTTP 500"):
        tts.synthesize("你好", voice_id="bad")
    assert tts.synthesize("你好", voice_id="fake").duration_ms == 400  # still serving


def test_the_sidecar_refuses_malformed_requests(sidecar_url: str) -> None:
    health = httpx.get(f"{sidecar_url}/health").json()
    assert health == {
        "status": "ok",
        "api_version": 1,
        "model": "fake",
        "sample_rate": 16000,
        "speakers": ["fake"],
    }
    for body in (
        {"speaker": "fake"},
        {"text": "你好"},
        {"text": "你好", "speaker": "fake", "prompt_audio": "AAAA"},
        {"text": "你好", "prompt_audio": "!!"},
        {"text": "你好", "speaker": "fake", "prompt_text": "x"},
        {"text": "你好", "speaker": "fake", "speed": 3},
    ):
        resp = httpx.post(f"{sidecar_url}/synthesize", json=body)
        assert resp.status_code == 400, body
        assert resp.json()["error"]
    assert httpx.post(f"{sidecar_url}/other", json={}).status_code == 404
