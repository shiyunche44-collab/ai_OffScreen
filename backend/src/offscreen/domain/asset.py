"""L0 Source: the movie file and what we derive from it."""

from __future__ import annotations

from pydantic import Field

from offscreen.domain.common import AssetId, Rational, Sha256, Strict, TimeMs, Versioned


class VideoInfo(Strict):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: Rational
    codec: str


class AudioStream(Strict):
    index: int = Field(ge=0)
    channels: int = Field(gt=0)
    sample_rate: int = Field(gt=0)
    language: str | None = None


class DerivedFiles(Strict):
    """Paths relative to data_dir."""

    proxy: str | None = None
    audio_16k: str | None = None
    audio_48k: str | None = None


class MediaAsset(Versioned):
    id: AssetId
    title: str
    source_path: str
    fingerprint: Sha256
    duration_ms: TimeMs
    video: VideoInfo
    audio: list[AudioStream]
    subtitles_external: str | None = None
    derived: DerivedFiles = DerivedFiles()
