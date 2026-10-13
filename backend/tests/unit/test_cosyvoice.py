"""The CosyVoice adapter against the sidecar's own HTTP app (sidecars/cosyvoice/server.py),
with a fake engine in place of the model."""

from __future__ import annotations

import base64
import importlib.util
import io
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from offscreen.config import AppConfig
from offscreen.domain.voice import Voice, VoiceLibrary
from offscreen.providers.adapters.cosyvoice_http import CosyVoiceTTS, VoiceReference
from offscreen.providers.adapters.tts_cache import CachedTTS
from offscreen.providers.ports import TTSError
from offscreen.services.tts import build_tts, library_references
from offscreen.store.voices import VoiceStore

SERVER = Path(__file__).resolve().parents[3] / "sidecars" / "cosyvoice" / "server.py"
MODEL = "CosyVoice2-0.5B"
RATE = 24_000


def load_server() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cosyvoice_sidecar", SERVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # pydantic resolves the request model's annotations here
    spec.loader.exec_module(module)
    return module


server = load_server()


def wav(seconds: float, rate: int = 16_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * round(seconds * rate))
    return buf.getvalue()


@dataclass
class FakeEngine:
    model: str = MODEL
    sample_rate: int = RATE
    calls: list[dict[str, Any]] = field(default_factory=list)
    fail: Exception | None = None

    def speakers(self) -> list[str]:
        return ["中文女"]

    def synthesize(
        self, text: str, *, speed: float, speaker: str | None, prompt: Any
    ) -> np.ndarray:
        self.calls.append({"text": text, "speed": speed, "speaker": speaker, "prompt": prompt})
        if self.fail is not None:
            raise self.fail
        if speaker is not None and speaker not in self.speakers():
            raise ValueError(f"unknown speaker {speaker!r}")
        seconds = len(text) / 5 / speed  # 5 characters a second
        return np.full(round(seconds * self.sample_rate), 0.25, dtype=np.float32)


def cfg_for(tmp_path: Path, **sidecar: Any) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data_dir": str(tmp_path),
            "asr": {"provider": "faster_whisper"},
            "tts": {"provider": "cosyvoice", "default_voice": "me"},
            "sidecars": {"cosyvoice": {"url": "http://sidecar.test", **sidecar}},
        }
    )


def through(app: Any) -> httpx.MockTransport:
    """Requests from the adapter's client go to the sidecar app in-process."""
    client = TestClient(app)

    def handle(req: httpx.Request) -> httpx.Response:
        r = client.request(req.method, req.url.path, content=req.content, headers=req.headers)
        return httpx.Response(r.status_code, headers=r.headers, content=r.content)

    return httpx.MockTransport(handle)


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


REF = VoiceReference(audio=wav(3.0), text="这是我的声音。")


def adapter(
    tmp_path: Path,
    engine: FakeEngine,
    refs: dict[str, VoiceReference] | None = None,
    **sidecar: Any,
) -> CosyVoiceTTS:
    table = {"me": REF} if refs is None else refs
    return CosyVoiceTTS(
        cfg_for(tmp_path, **sidecar),
        table.get,
        transport=through(server.create_app(engine)),
    )


# ---- the sidecar on its own ------------------------------------------------------------------
def test_sidecar_health_and_validation(engine: FakeEngine) -> None:
    client = TestClient(server.create_app(engine))
    assert client.get("/health").json() == {
        "model": MODEL,
        "sample_rate": RATE,
        "speakers": ["中文女"],
    }
    both = {"text": "你好", "speaker": "中文女", "prompt_audio": "AAAA"}
    assert client.post("/tts", json=both).status_code == 422
    assert client.post("/tts", json={"text": "你好"}).status_code == 422
    assert client.post("/tts", json={"text": "你好", "speaker": "x", "speed": 3}).status_code == 422
    bad = client.post("/tts", json={"text": "你好", "prompt_audio": "bm90IGEgd2F2"})
    assert bad.status_code == 400 and "WAV" in bad.json()["detail"]
    assert engine.calls == []


def test_sidecar_returns_16_bit_mono_wav(engine: FakeEngine) -> None:
    client = TestClient(server.create_app(engine))
    body = {"text": "你好世界啊", "prompt_audio": base64.b64encode(wav(1)).decode()}
    r = client.post("/tts", json=body)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
        assert w.getnframes() == RATE  # 5 characters at 5 a second
    prompt = engine.calls[0]["prompt"]
    assert prompt.text is None and prompt.audio == wav(1)  # cross-lingual: no transcript


# ---- adapter <-> sidecar -------------------------------------------------------------------
def test_a_library_voice_is_cloned_from_its_recording(tmp_path: Path, engine: FakeEngine) -> None:
    tts = adapter(tmp_path, engine)
    out = tts.synthesize("今天天气很好", voice_id="me", speed=1.5)
    (call,) = engine.calls
    assert call["text"] == "今天天气很好" and call["speed"] == 1.5 and call["speaker"] is None
    assert call["prompt"].audio == REF.audio and call["prompt"].text == REF.text
    assert (out.format, out.sample_rate, out.duration_ms) == ("wav", RATE, 800)
    assert out.char_timings == [] and out.billed_chars is None
    assert tts.id == f"cosyvoice:{MODEL}:wav"


def test_other_voice_ids_are_built_in_speakers(tmp_path: Path, engine: FakeEngine) -> None:
    tts = adapter(tmp_path, engine)
    tts.synthesize("你好", voice_id="中文女")
    assert engine.calls[0]["speaker"] == "中文女" and engine.calls[0]["prompt"] is None
    with pytest.raises(TTSError, match="HTTP 400: unknown speaker"):
        tts.synthesize("你好", voice_id="nobody")


def test_inference_failure_is_reported(tmp_path: Path, engine: FakeEngine) -> None:
    engine.fail = RuntimeError("CUDA out of memory")
    with pytest.raises(TTSError, match="HTTP 500: inference failed: RuntimeError: CUDA out"):
        adapter(tmp_path, engine).synthesize("你好", voice_id="me")


def test_the_sidecar_must_run_the_configured_model(tmp_path: Path, engine: FakeEngine) -> None:
    tts = adapter(tmp_path, engine, model="CosyVoice-300M-SFT")
    with pytest.raises(TTSError, match=r"runs 'CosyVoice2-0\.5B'"):
        tts.synthesize("你好", voice_id="me")
    assert engine.calls == []


def test_a_sidecar_that_is_down_says_how_to_start_it(tmp_path: Path) -> None:
    def refuse(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=req)

    tts = CosyVoiceTTS(cfg_for(tmp_path), {}.get, transport=httpx.MockTransport(refuse))
    with pytest.raises(TTSError, match=r"not reachable at http://sidecar.test; .*README"):
        tts.synthesize("你好", voice_id="me")


def test_bad_requests_never_reach_the_sidecar(tmp_path: Path, engine: FakeEngine) -> None:
    tts = adapter(tmp_path, engine)
    for text, speed in (("  ", 1.0), ("你好", 0.3), ("字" * 10_001, 1.0)):
        with pytest.raises(ValueError):
            tts.synthesize(text, voice_id="me", speed=speed)
    assert engine.calls == []


# ---- cache and wiring ----------------------------------------------------------------------
def test_a_new_recording_is_a_new_voice_for_the_cache(tmp_path: Path, engine: FakeEngine) -> None:
    refs = {"me": REF}
    inner = adapter(tmp_path, engine, refs)
    cached = CachedTTS(inner, tmp_path / "cache")
    cached.synthesize("你好", voice_id="me")
    cached.synthesize("你好", voice_id="me")
    assert (cached.hits, cached.misses) == (1, 1)

    refs["me"] = VoiceReference(audio=wav(4.0), text=REF.text)
    cached.synthesize("你好", voice_id="me")
    refs["me"] = VoiceReference(audio=wav(4.0), text="另一句话。")
    cached.synthesize("你好", voice_id="me")
    assert (cached.hits, cached.misses) == (1, 3) and len(engine.calls) == 3


def test_build_tts_clones_from_the_voice_library(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path)
    store.dir.mkdir(parents=True)
    (store.dir / "me.wav").write_bytes(REF.audio)
    store.write(
        VoiceLibrary(
            voices=[
                Voice(
                    id="me",
                    name="我",
                    provider="cosyvoice",
                    reference_audio="me.wav",
                    reference_text=REF.text,
                ),
                Voice(id="plain", name="plain", provider="cosyvoice"),
            ]
        )
    )
    refs = library_references(store)
    assert refs("me") == REF
    assert refs("plain") is None and refs("unknown") is None
    (store.dir / "me.wav").unlink()
    with pytest.raises(TTSError, match="cannot read its recording"):
        refs("me")

    tts = build_tts(cfg_for(tmp_path))
    assert isinstance(tts.inner, CosyVoiceTTS) and tts.dir == tmp_path / "tts_cache"
