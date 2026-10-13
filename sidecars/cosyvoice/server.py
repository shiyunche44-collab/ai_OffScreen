"""CosyVoice sidecar: speech synthesis over HTTP, in its own environment (ARCHITECTURE §3, ADR 011).

No business logic: text + voice in, WAV out. The platform's adapter is
`backend/src/offscreen/providers/adapters/cosyvoice_tts.py`; the contract is in README.md.

Only the standard library is needed to serve; CosyVoice itself (torch, the model) is imported
when the real engine starts, so `--engine fake` runs anywhere (contract tests).

    python server.py --model-dir pretrained_models/CosyVoice2-0.5B [--host 127.0.0.1] [--port 50021]
    python server.py --engine fake --port 0      # prints the port it listens on
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import sys
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

API_VERSION = 1
MAX_TEXT_CHARS = 2_000
MAX_BODY_BYTES = 32 * 1024 * 1024
SPEED_RANGE = (0.5, 2.0)


class BadRequest(ValueError):
    pass


class Engine(Protocol):
    model: str
    sample_rate: int

    def speakers(self) -> list[str]: ...

    def synthesize(
        self,
        text: str,
        *,
        speaker: str | None,
        prompt_wav: Path | None,
        prompt_text: str | None,
        speed: float,
    ) -> bytes:
        """16-bit mono PCM at `sample_rate`."""
        ...


class CosyVoiceEngine:
    """CosyVoice 2 (or 1) through its own Python API. Cloning: `zero_shot` when the
    transcript of the reference recording is known, `cross_lingual` (no transcript) otherwise;
    a built-in speaker uses `sft`."""

    def __init__(self, model_dir: str, *, fp16: bool) -> None:
        import torch  # noqa: F401  (fails early with a clear message when the env is wrong)
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2  # type: ignore[import-not-found]
        from cosyvoice.utils.file_utils import load_wav  # type: ignore[import-not-found]

        self._load_wav = load_wav
        try:
            self._cv = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=fp16)
        except Exception:  # a CosyVoice 1 model directory
            self._cv = CosyVoice(model_dir, load_jit=False, load_trt=False, fp16=fp16)
        self.model = Path(model_dir).name
        self.sample_rate = int(self._cv.sample_rate)

    def speakers(self) -> list[str]:
        return list(self._cv.list_available_spks())

    def synthesize(
        self,
        text: str,
        *,
        speaker: str | None,
        prompt_wav: Path | None,
        prompt_text: str | None,
        speed: float,
    ) -> bytes:
        import numpy as np
        import torch

        if prompt_wav is not None:
            prompt = self._load_wav(str(prompt_wav), 16000)
            if prompt_text:
                chunks = self._cv.inference_zero_shot(
                    text, prompt_text, prompt, stream=False, speed=speed
                )
            else:
                chunks = self._cv.inference_cross_lingual(text, prompt, stream=False, speed=speed)
        else:
            if speaker not in self.speakers():
                raise BadRequest(f"unknown speaker {speaker!r}; this model has {self.speakers()}")
            chunks = self._cv.inference_sft(text, speaker, stream=False, speed=speed)
        speech = torch.cat([c["tts_speech"] for c in chunks], dim=1).squeeze(0).cpu().numpy()
        pcm = (np.clip(speech, -1.0, 1.0) * 32767).astype("<i2")
        return bytes(pcm.tobytes())


class FakeEngine:
    """Silence whose length follows the text (5 chars/s, scaled by speed). For contract tests."""

    model = "fake"
    sample_rate = 16000

    def speakers(self) -> list[str]:
        return ["fake"]

    def synthesize(
        self,
        text: str,
        *,
        speaker: str | None,
        prompt_wav: Path | None,
        prompt_text: str | None,
        speed: float,
    ) -> bytes:
        if prompt_wav is None and speaker not in self.speakers():
            raise BadRequest(f"unknown speaker {speaker!r}; this model has {self.speakers()}")
        if prompt_wav is not None:
            with wave.open(str(prompt_wav), "rb"):
                pass  # an unreadable recording fails like the real engine would
        spoken = sum(1 for c in text if not c.isspace())
        n = round(spoken / 5.0 / speed * self.sample_rate)
        return b"\x00\x00" * max(n, 1)


def to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def parse_request(body: bytes) -> dict[str, Any]:
    try:
        req = json.loads(body)
    except ValueError as e:
        raise BadRequest("body is not JSON") from e
    if not isinstance(req, dict):
        raise BadRequest("body must be a JSON object")
    text = req.get("text")
    if not isinstance(text, str) or not text.strip():
        raise BadRequest("text is required")
    if len(text) > MAX_TEXT_CHARS:
        raise BadRequest(f"text is {len(text)} chars; the limit is {MAX_TEXT_CHARS}")
    speed = req.get("speed", 1.0)
    if not isinstance(speed, int | float) or not SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
        raise BadRequest(f"speed must be within {SPEED_RANGE}")
    speaker, prompt_audio, prompt_text = (
        req.get("speaker"),
        req.get("prompt_audio"),
        req.get("prompt_text"),
    )
    if (speaker is None) == (prompt_audio is None):
        raise BadRequest("give exactly one of speaker (built-in voice) and prompt_audio (clone)")
    prompt: bytes | None = None
    if prompt_audio is not None:
        try:
            prompt = base64.b64decode(prompt_audio, validate=True)
        except (binascii.Error, TypeError) as e:
            raise BadRequest("prompt_audio is not base64") from e
    if prompt_text is not None and (not isinstance(prompt_text, str) or prompt is None):
        raise BadRequest("prompt_text goes with prompt_audio")
    return {
        "text": text,
        "speed": float(speed),
        "speaker": speaker,
        "prompt": prompt,
        "prompt_text": prompt_text or None,
    }


def make_handler(engine: Engine) -> type[BaseHTTPRequestHandler]:
    gpu = threading.Lock()  # one synthesis at a time; the others wait

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # requests only, never bodies
            sys.stderr.write(f"cosyvoice-sidecar {self.address_string()} {fmt % args}\n")

        def do_GET(self) -> None:
            if self.path != "/health":
                self._json(404, {"error": "not found"})
                return
            self._json(
                200,
                {
                    "status": "ok",
                    "api_version": API_VERSION,
                    "model": engine.model,
                    "sample_rate": engine.sample_rate,
                    "speakers": engine.speakers(),
                },
            )

        def do_POST(self) -> None:
            if self.path != "/synthesize":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._json(400, {"error": f"body must be 1..{MAX_BODY_BYTES} bytes"})
                return
            body = self.rfile.read(length)
            try:
                req = parse_request(body)
                with gpu, tempfile.TemporaryDirectory() as tmp:
                    prompt_wav = None
                    if req["prompt"] is not None:
                        prompt_wav = Path(tmp) / "prompt.wav"
                        prompt_wav.write_bytes(req["prompt"])
                    pcm = engine.synthesize(
                        req["text"],
                        speaker=req["speaker"],
                        prompt_wav=prompt_wav,
                        prompt_text=req["prompt_text"],
                        speed=req["speed"],
                    )
            except BadRequest as e:
                self._json(400, {"error": str(e)})
                return
            except Exception as e:  # a failed inference; the server keeps running
                self._json(500, {"error": f"{type(e).__name__}: {e}"[:500]})
                return
            if not pcm:
                self._json(500, {"error": "the engine produced no audio"})
                return
            self._send(200, "audio/wav", to_wav(pcm, engine.sample_rate))

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(status, "application/json", json.dumps(payload, ensure_ascii=False).encode())

        def _send(self, status: int, ctype: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--engine", choices=["cosyvoice", "fake"], default="cosyvoice")
    ap.add_argument("--model-dir", default=os.environ.get("COSYVOICE_MODEL_DIR"))
    ap.add_argument("--fp16", action="store_true", help="half precision (GPU)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50021)
    args = ap.parse_args(argv)
    if args.engine == "fake":
        engine: Engine = FakeEngine()
    else:
        if not args.model_dir:
            ap.error("--model-dir (or COSYVOICE_MODEL_DIR) is required")
        engine = CosyVoiceEngine(args.model_dir, fp16=args.fp16)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"listening on {args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
