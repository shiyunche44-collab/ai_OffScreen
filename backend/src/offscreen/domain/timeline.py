"""L4 Timeline: absolute, frame-aligned tracks. Produced only by the compiler;
the renderer's single input. Output positions are integer frames, half-open [f0, f1)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from offscreen.domain.common import AssetId, Frame, Rational, SegmentId, Strict, TimeMs, Versioned
from offscreen.domain.plan import SourceAudio  # noqa: F401  (re-exported for compilers)


class PlanRef(Strict):
    id: str
    version: int = Field(ge=1)


class OutputSpec(Strict):
    profile: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: Rational
    layout: Literal["keep", "blur_pad", "center_crop"] = "keep"


class _Span(Strict):
    f0: Frame
    f1: Frame

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.f0 >= self.f1:
            raise ValueError("f0 must be < f1")
        return self


class VideoItem(_Span):
    seg: SegmentId
    asset_id: AssetId
    src_in_ms: TimeMs
    src_out_ms: TimeMs
    speed: float = Field(gt=0, default=1.0)


class NarrationItem(Strict):
    seg: SegmentId
    f0: Frame
    file: str
    gain_db: float = 0.0


class SourceAudioItem(_Span):
    seg: SegmentId
    asset_id: AssetId
    stem: Literal["mix", "no_vocals", "vocals"]
    src_in_ms: TimeMs
    gain_db: float = 0.0


class Duck(Strict):
    under: Literal["narration"] = "narration"
    depth_db: float = -8.0


class BgmItem(_Span):
    file: str
    gain_db: float = -22.0
    duck: Duck | None = None


class SubtitleItem(_Span):
    text: str


class OverlayItem(_Span):
    type: Literal["title"]
    text: str


class Timeline(Versioned):
    plan_ref: PlanRef
    output: OutputSpec
    duration_frames: int = Field(gt=0)
    video: list[VideoItem]
    narration: list[NarrationItem] = []
    source_audio: list[SourceAudioItem] = []
    bgm: list[BgmItem] = []
    subtitles: list[SubtitleItem] = []
    overlays: list[OverlayItem] = []

    @model_validator(mode="after")
    def _frame_grid(self) -> Self:
        # The video track must tile [0, duration_frames) exactly: no gaps, no overlap.
        cursor = 0
        for item in self.video:
            if item.f0 != cursor:
                raise ValueError(f"video track gap/overlap at frame {item.f0} (expected {cursor})")
            cursor = item.f1
        if cursor != self.duration_frames:
            raise ValueError(f"video ends at {cursor}, duration_frames is {self.duration_frames}")
        for track in (self.source_audio, self.bgm, self.subtitles, self.overlays):
            for it in track:
                if it.f1 > self.duration_frames:
                    raise ValueError("item extends past duration_frames")
        for n in self.narration:
            if n.f0 >= self.duration_frames:
                raise ValueError("narration starts past duration_frames")
        return self
