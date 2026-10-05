"""Provider ports: interfaces the stages depend on. Vendor SDKs live in adapters/ only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

from offscreen.domain.common import TimeRange
from offscreen.domain.index import Word
from offscreen.domain.llm import LlmCallRecord


class DetectionCanceled(RuntimeError):
    """Raised by a detector when `should_cancel` turned true mid-run."""


class ShotDetector(Protocol):
    """Finds hard cuts in a video."""

    @property
    def id(self) -> str:
        """Identifies the algorithm and its settings; part of the stage's cache key."""
        ...

    def detect(
        self,
        video: Path,
        *,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[TimeRange]:
        """Consecutive shots in source order, as raw detector output. They need not cover
        the whole video; `algo.shots.normalize_shots` makes them a clean partition.
        Raises DetectionCanceled if `should_cancel` returns true."""
        ...


class AsrCanceled(RuntimeError):
    """Raised by an ASR provider when `should_cancel` turned true mid-run."""


@dataclass(frozen=True)
class AsrSegment:
    """One recognized utterance. Ids are assigned later, by the stage."""

    start_ms: int
    end_ms: int
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass(frozen=True)
class AsrResult:
    language: str
    segments: list[AsrSegment]


class ASR(Protocol):
    @property
    def id(self) -> str:
        """Engine, model and settings; part of the stage's cache key."""
        ...

    def transcribe(
        self,
        wav: Path,
        *,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AsrResult:
        """Recognize speech in a 16 kHz mono wav, with word timestamps when the engine has
        them. Raises AsrCanceled if `should_cancel` returns true."""
        ...


@dataclass(frozen=True)
class DetectedFace:
    bbox: tuple[float, float, float, float]
    """x0, y0, x1, y1 as fractions of the image (0..1)."""
    score: float
    embedding: tuple[float, ...]
    """Identity feature; faces of one person are close under cosine similarity."""


class FaceAnalyzer(Protocol):
    @property
    def id(self) -> str:
        """Engine, model and settings; part of the stage's cache key."""
        ...

    def detect_and_embed(self, image: Path) -> list[DetectedFace]:
        """Every face in the image file, with its feature vector. May be empty. Raises
        RuntimeError when the engine is not installed or the image cannot be read."""
        ...


# ---- LLM ---------------------------------------------------------------------------------
M = TypeVar("M", bound=BaseModel)


class LLMError(RuntimeError):
    """A call failed and retrying the same request will not help (or retries ran out)."""


class LLMAuthError(LLMError):
    """Missing, invalid or unauthorized key."""


class LLMQuotaExhausted(LLMError):
    """Balance or subscription window used up. Not retried: waiting minutes will not fix it."""

    def __init__(self, message: str, *, provider: str = "", reset_hint: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.reset_hint = reset_hint


class LLMRateLimited(LLMError):
    """Still rate limited (or a 5xx / network failure) after the backoff retries."""


class LLMSchemaError(LLMError):
    """The reply did not match the schema, even after one repair attempt."""


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    images: tuple[str, ...] = ()
    """`data:image/...;base64,...` URLs attached to this message (vision models only)."""


Recorder = Callable[[LlmCallRecord], None]


class LLM(Protocol):
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
        """Run `task` (its provider and model come from configuration) and return a reply
        that passed `schema` validation. Raises LLMError subclasses; never returns
        unvalidated data."""
        ...


# ---- TTS ---------------------------------------------------------------------------------
class TTSError(RuntimeError):
    """Synthesis failed and retrying the same request will not help (or retries ran out)."""


class TTSAuthError(TTSError):
    """Missing, invalid or unauthorized key."""


class TTSQuotaExhausted(TTSError):
    """Balance or subscription window used up. Not retried."""

    def __init__(self, message: str, *, provider: str = "", reset_hint: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.reset_hint = reset_hint


class TTSRateLimited(TTSError):
    """Still rate limited (or a 5xx / network failure) after the backoff retries."""


@dataclass(frozen=True)
class SynthesizedAudio:
    data: bytes
    format: str
    """Container of `data`, e.g. "mp3"."""
    sample_rate: int
    duration_ms: int
    char_timings: list[tuple[int, int]]
    """`(start_ms, end_ms)` inside the audio for each character of the input text, or an
    empty list when the engine gives no timestamps. Never partial: all or nothing."""
    billed_chars: int | None = None


class TTS(Protocol):
    @property
    def id(self) -> str:
        """Engine, model and audio settings; part of the cache key of anything synthesized."""
        ...

    def synthesize(self, text: str, *, voice_id: str, speed: float = 1.0) -> SynthesizedAudio:
        """Speak `text`. Raises TTSError subclasses; never returns empty audio."""
        ...
