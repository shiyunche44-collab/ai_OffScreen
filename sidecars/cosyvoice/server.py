"""CosyVoice sidecar: speech synthesis with voice cloning over HTTP (ARCHITECTURE §3.2).

Python 3.10 (what CosyVoice requires): no `typing.Self` and the like.

Runs in its own environment (CosyVoice pins torch and friends; see README.md) and only does
"input → inference → output": no projects, no caching, no knowledge of the voice library. The
caller sends the reference recording with every request.

    GET  /health -> {"model": str, "sample_rate": int, "speakers": [str]}
    POST /tts    {"text", "speed", and either "speaker" (a built-in voice of an SFT model)
                  or "prompt_audio" (base64 WAV) with an optional "prompt_text"}
                 -> audio/wav, 16-bit mono

Errors: 400 for a request the model cannot serve (bad audio, unknown speaker), 500 when
inference itself fails; the body is {"detail": "..."}. One synthesis runs at a time (one GPU).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import tempfile
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator

MAX_TEXT_CHARS = 10_000


@dataclass(frozen=True)
class Prompt:
    audio: bytes
    """A WAV file."""
    text: str | None
    """What is said in it; None for cross-lingual cloning."""


class Engine(Protocol):
    model: str
    sample_rate: int

    def speakers(self) -> list[str]: ...

    def synthesize(
        self, text: str, *, speed: float, speaker: str | None, prompt: Prompt | None
    ) -> np.ndarray:
        """Mono float samples in [-1, 1] at `sample_rate`. ValueError: the request cannot be
        served (it is the caller's to fix)."""
        ...


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    speaker: str | None = None
    prompt_audio: str | None = None
    """Base64 of a WAV file."""
    prompt_text: str | None = None

    @model_validator(mode="after")
    def _one_voice(self) -> TtsRequest:
        if (self.speaker is None) == (self.prompt_audio is None):
            raise ValueError("give either speaker or prompt_audio")
        if self.prompt_text is not None and self.prompt_audio is None:
            raise ValueError("prompt_text needs prompt_audio")
        return self


def to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype(
        "<i2"
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _prompt(req: TtsRequest) -> Prompt | None:
    if req.prompt_audio is None:
        return None
    try:
        audio = base64.b64decode(req.prompt_audio, validate=True)
        with wave.open(io.BytesIO(audio)) as w:
            if w.getnframes() == 0:
                raise ValueError("the reference recording is empty")
    except (binascii.Error, wave.Error, EOFError, ValueError) as e:
        raise HTTPException(400, f"prompt_audio is not a readable WAV file: {e}") from e
    return Prompt(audio=audio, text=req.prompt_text or None)


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="cosyvoice sidecar")
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "model": engine.model,
            "sample_rate": engine.sample_rate,
            "speakers": engine.speakers(),
        }

    @app.post("/tts")
    def tts(req: TtsRequest) -> Response:
        prompt = _prompt(req)
        with lock:
            try:
                samples = engine.synthesize(
                    req.text, speed=req.speed, speaker=req.speaker, prompt=prompt
                )
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            except Exception as e:
                raise HTTPException(500, f"inference failed: {type(e).__name__}: {e}") from e
        if np.asarray(samples).size == 0:
            raise HTTPException(500, "inference produced no audio")
        return Response(to_wav(samples, engine.sample_rate), media_type="audio/wav")

    return app


class CosyVoiceEngine:
    """The real model. Written against CosyVoice2 (`cosyvoice.cli.cosyvoice.CosyVoice2`,
    `inference_zero_shot / inference_cross_lingual / inference_sft`, prompt audio as a 16 kHz
    tensor from `load_wav`); CosyVoice-300M-SFT models load with `CosyVoice`."""

    def __init__(self, model_dir: Path, *, fp16: bool = False) -> None:
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2  # the sidecar's env only
        from cosyvoice.utils.file_utils import load_wav

        cls = CosyVoice2 if "CosyVoice2" in model_dir.name else CosyVoice
        self._model = cls(str(model_dir), load_jit=False, load_trt=False, fp16=fp16)
        self._load_wav = load_wav
        self.model = model_dir.name
        self.sample_rate = int(self._model.sample_rate)

    def speakers(self) -> list[str]:
        return [str(s) for s in self._model.list_available_spks()]

    def synthesize(
        self, text: str, *, speed: float, speaker: str | None, prompt: Prompt | None
    ) -> np.ndarray:
        if prompt is None:
            if speaker not in self.speakers():
                raise ValueError(f"unknown speaker {speaker!r}; this model has {self.speakers()}")
            chunks = self._model.inference_sft(text, speaker, stream=False, speed=speed)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                f.write(prompt.audio)
                f.flush()
                speech_16k = self._load_wav(f.name, 16000)
            if prompt.text:
                chunks = self._model.inference_zero_shot(
                    text, prompt.text, speech_16k, stream=False, speed=speed
                )
            else:
                chunks = self._model.inference_cross_lingual(
                    text, speech_16k, stream=False, speed=speed
                )
        parts = [c["tts_speech"].squeeze(0).float().cpu().numpy() for c in chunks]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50021)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    import uvicorn

    engine = CosyVoiceEngine(args.model_dir, fp16=args.fp16)  # loads before the port opens
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
