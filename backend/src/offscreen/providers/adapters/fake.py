"""Scripted LLM for tests and offline runs: no network, deterministic, inspectable."""

from __future__ import annotations

import io
import wave
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from offscreen.algo.jsonreply import extract_json
from offscreen.algo.tts import WordSpan, char_timings_from_spans
from offscreen.domain.llm import LlmCallRecord
from offscreen.providers.ports import (
    DetectedFace,
    LLMError,
    M,
    Message,
    Recorder,
    SynthesizedAudio,
)

Reply = BaseModel | dict[str, Any] | str | Exception
"""A model instance or dict (validated against the requested schema), a string (parsed
like a real reply, so fences and prose are exercised), or an exception to raise."""
ReplyFn = Callable[[str, list[Message], type[BaseModel]], Reply]


class FakeLLM:
    """`replies[task]` is a list used in order, or a function of (task, messages, schema)."""

    def __init__(
        self,
        replies: Mapping[str, list[Reply] | ReplyFn],
        recorder: Recorder | None = None,
    ) -> None:
        self._queues: dict[str, deque[Reply]] = defaultdict(deque)
        self._fns: dict[str, ReplyFn] = {}
        for task, r in replies.items():
            if callable(r):
                self._fns[task] = r
            else:
                self._queues[task].extend(r)
        self._recorder = recorder
        self.calls: list[tuple[str, list[Message], str]] = []
        """(task, messages, prompt_version) for every generate() call."""

    def generate(
        self,
        task: str,
        messages: list[Message],
        schema: type[M],
        *,
        prompt_version: str,
        job_id: str | None = None,
        max_tokens: int = 4096,
    ) -> M:
        self.calls.append((task, list(messages), prompt_version))
        if task in self._fns:
            reply = self._fns[task](task, messages, schema)
        elif self._queues[task]:
            reply = self._queues[task].popleft()
        else:
            raise LLMError(f"FakeLLM: no scripted reply left for task {task!r}")
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            value = schema.model_validate(extract_json(reply))
        elif isinstance(reply, BaseModel):
            value = schema.model_validate(reply.model_dump())
        else:
            value = schema.model_validate(reply)
        if self._recorder is not None:
            self._recorder(
                LlmCallRecord(
                    task=task,
                    provider="fake",
                    model="fake",
                    prompt_version=prompt_version,
                    status="ok",
                    job_id=job_id,
                    request={
                        "messages": [{"role": m.role, "content": m.content} for m in messages]
                    },
                    response={"content": value.model_dump_json()},
                )
            )
        return value


class FakeTTS:
    """Offline TTS: a silent wav whose length follows the text (`chars_per_s`, scaled by
    `speed`), with evenly spread character timings. Deterministic; records its calls."""

    def __init__(
        self,
        *,
        chars_per_s: float = 4.5,
        sample_rate: int = 16000,
        fail_with: Exception | None = None,
    ) -> None:
        self.chars_per_s = chars_per_s
        self.sample_rate = sample_rate
        self.fail_with = fail_with
        self.calls: list[tuple[str, str, float]] = []
        """(text, voice_id, speed) for every synthesize() call."""

    @property
    def id(self) -> str:
        return f"fake-tts:wav@{self.sample_rate}:{self.chars_per_s}"

    def synthesize(self, text: str, *, voice_id: str, speed: float = 1.0) -> SynthesizedAudio:
        self.calls.append((text, voice_id, speed))
        if self.fail_with is not None:
            raise self.fail_with
        spoken = [i for i, c in enumerate(text) if not c.isspace()]
        if not spoken:
            raise ValueError("cannot synthesize empty text")
        duration_ms = max(1, round(len(spoken) / self.chars_per_s / speed * 1000))
        n_samples = round(duration_ms * self.sample_rate / 1000)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(b"\x00\x00" * n_samples)
        step = duration_ms / len(spoken)
        spans = [WordSpan(k * step, (k + 1) * step, i, i + 1) for k, i in enumerate(spoken)]
        return SynthesizedAudio(
            data=buf.getvalue(),
            format="wav",
            sample_rate=self.sample_rate,
            duration_ms=duration_ms,
            char_timings=char_timings_from_spans(len(text), spans, duration_ms),
            billed_chars=len(spoken),
        )


class FakeFaceAnalyzer:
    """Offline face analysis: `faces_in(image path)` says what is found (default: nothing).
    Records the images it was asked about."""

    id = "fake-faces@1"

    def __init__(self, faces_in: Callable[[Path], list[DetectedFace]] | None = None) -> None:
        self._faces_in = faces_in or (lambda _p: [])
        self.images: list[Path] = []

    def detect_and_embed(self, image: Path) -> list[DetectedFace]:
        self.images.append(image)
        return self._faces_in(image)


class FakeEmbedder:
    """Offline embeddings: texts become a hashed bag of character 1-3-grams (so texts that share
    words are close, deterministic); images come from `image_of(path)` (default: from the file's
    bytes). Records what it was asked to embed."""

    def __init__(
        self,
        dim: int = 256,
        *,
        image_of: Callable[[Path], str] | None = None,
        id: str = "fake-embedder@1",
        images: bool = True,
    ) -> None:
        self._dim = dim
        self._image_of = image_of
        self.id = id
        self._images = images
        self.image_calls: list[list[Path]] = []
        self.text_calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        import hashlib
        import math

        v = [0.0] * self._dim
        grams = [text[i : i + n] for n in (1, 2, 3) for i in range(max(1, len(text) - n + 1))]
        for g in grams:
            h = int.from_bytes(hashlib.sha256(g.encode()).digest()[:4], "big")
            v[h % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.text_calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def embed_images(self, images: Sequence[Path]) -> list[list[float]]:
        if not self._images:
            raise RuntimeError("this embedder embeds text only")
        self.image_calls.append(list(images))
        of = self._image_of or (lambda p: p.read_bytes().hex())
        return [self._vector(of(p)) for p in images]
