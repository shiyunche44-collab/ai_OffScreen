"""How fast a voice speaks, measured on standard texts, and what that says about the length of a
text. Pure: the synthesis happens in the voice service."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from offscreen.algo.script import DEFAULT_CHARS_PER_S, count_chars
from offscreen.domain.voice import Voice

STANDARD_TEXTS: tuple[str, ...] = (
    "这座小镇已经很多年没有下过雪了。可是那天夜里，人们醒来的时候，发现街道、屋顶和远处的山"
    "都被一层薄薄的白色盖住。没有人知道这场雪是从哪里来的，更没有人知道，它将带走什么。",
    "男孩第一次见到那只小龙，是在一个安静的清晨。它蜷缩在河边的石头旁边，翅膀受了伤，眼睛却"
    "一直盯着他。他犹豫了很久，还是把它抱回了家，从此，他的生活再也不一样了。",
    "接下来的故事发生得很快。一场意外，一次逃亡，几个陌生人的出现，让原本平静的日子彻底被打"
    "乱。直到最后一刻，他才明白，自己真正要守护的，从来都不是眼前这座城。",
)
"""Narration-like paragraphs: sentences of varied length, commas and full stops, no digits or
Latin letters (how those are read is a separate matter, M6-02)."""

TARGET_SPREAD = 0.05
"""The estimate of a text's length is meant to be within this of the audio's."""


@dataclass(frozen=True)
class RateFit:
    chars_per_s: float
    spread: float
    """The largest relative deviation of a sample's own rate from `chars_per_s`."""


def fit_rate(samples: Sequence[tuple[int, int]]) -> RateFit:
    """The speaking rate from `(spoken characters, audio duration in ms)` samples taken at
    speed 1.0: all characters over all seconds, so a long text counts for more than a short one."""
    if not samples:
        raise ValueError("no samples")
    if any(c < 1 or ms < 1 for c, ms in samples):
        raise ValueError("every sample needs characters and a duration")
    rate = sum(c for c, _ in samples) / (sum(ms for _, ms in samples) / 1000)
    spread = max(abs(c / (ms / 1000) - rate) / rate for c, ms in samples)
    return RateFit(rate, spread)


def sample_of(text: str, duration_ms: int) -> tuple[int, int]:
    return count_chars(text), duration_ms


@dataclass(frozen=True)
class VoiceTiming:
    chars_per_s: float
    """Spoken characters per second at the speed plans use for the voice."""
    speed: float
    """That speed."""
    measured: bool
    """False: the default rate, nobody has measured this voice (with this engine)."""


def voice_timing(voice: Voice | None, tts_id: str) -> VoiceTiming:
    """What to assume of a voice when estimating length: its measured rate times its default
    speed, if it was measured with this very engine; else the default rate at the voice's speed."""
    speed = voice.default_speed if voice else 1.0
    if voice and voice.chars_per_s is not None and voice.calibrated_with == tts_id:
        return VoiceTiming(voice.chars_per_s * speed, speed, True)
    return VoiceTiming(DEFAULT_CHARS_PER_S * speed, speed, False)
