"""Programme mixdown (ARCHITECTURE §7.5 ②, ADR 012): every sound placed sample by sample on one
buffer, with its gain, short fades at its edges and, for music, a ducking envelope under the
narration. Pure numpy: decoding and writing files is `media/audio.py`.

Audio is float32, shape `(samples, channels)`, nominally within [-1, 1]. Positions are
integer sample indices of the programme; gains are in dB."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from offscreen.domain.common import Rational

Audio = NDArray[np.float32]


@dataclass(frozen=True)
class MixSettings:
    sample_rate: int = 48_000
    channels: int = 2
    edge_fade_ms: int = 15
    """Fade in and out at both ends of every placed piece (10–20 ms): a cut between two pieces
    of the film's sound, or a piece starting mid-waveform, then does not click."""
    duck_attack_ms: int = 150
    """How long before the narration starts the music begins to go down (the mix is offline,
    so the voice can start on music that is already low)."""
    duck_release_ms: int = 400
    """How long after the narration ends the music takes to come back."""

    def __post_init__(self) -> None:
        if not 10 <= self.edge_fade_ms <= 20:
            raise ValueError("edge_fade_ms must be within 10..20")
        if self.duck_attack_ms < 0 or self.duck_release_ms < 0:
            raise ValueError("duck attack and release must be >= 0")
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("sample_rate and channels must be positive")

    def samples(self, ms: int | float) -> int:
        return round(ms * self.sample_rate / 1000)


@dataclass(frozen=True)
class Placement:
    """`audio` starts at programme sample `start` (may be cut by either end of the programme)."""

    audio: Audio
    start: int
    gain_db: float = 0.0


@dataclass(frozen=True)
class Layer:
    """Pieces mixed the same way; `duck_db` lowers the whole layer while `duck_under` sounds."""

    placements: Sequence[Placement]
    duck_db: float | None = None
    duck_under: Sequence[tuple[int, int]] = field(default_factory=tuple)
    """Half-open sample spans of the sound this layer gives way to (the narration)."""


@dataclass(frozen=True)
class MixReport:
    peak: float
    """Largest absolute sample value (1.0 is full scale)."""
    clipped: int
    """Samples beyond full scale (clamped when written as 16-bit)."""


def frame_to_sample(frame: int, fps: Rational, sample_rate: int) -> int:
    """The programme sample at which output frame `frame` starts."""
    return round(frame * sample_rate * fps.den / fps.num)


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


def edge_fades(n: int, fade: int) -> NDArray[np.float32]:
    """Per-sample gain of a piece `n` samples long: raised-cosine ramps of `fade` samples at
    both ends (shorter when the piece is shorter than two fades), 1 in between."""
    gain = np.ones(n, dtype=np.float32)
    k = min(fade, n // 2)
    if k > 0:
        ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(k, dtype=np.float64) + 0.5) / k)).astype(
            np.float32
        )
        gain[:k] = ramp
        gain[n - k :] = ramp[::-1]
    return gain


def merge_spans(spans: Sequence[tuple[int, int]], bridge: int = 0) -> list[tuple[int, int]]:
    """Sorted, non-empty spans with overlaps joined, and gaps shorter than `bridge` too."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if out and a - out[-1][1] < bridge:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def duck_envelope(
    total: int,
    spans: Sequence[tuple[int, int]],
    depth_db: float,
    attack: int,
    release: int,
) -> NDArray[np.float32]:
    """Per-sample gain (linear) of a layer ducked by `depth_db` (<= 0) during `spans`.

    The level moves linearly in dB: down over the `attack` samples before each span, back up
    over the `release` samples after it. Spans closer together than attack + release are
    treated as one, so the music does not pump between two sentences."""
    db = np.zeros(total, dtype=np.float32)
    if depth_db >= 0 or total <= 0:
        return np.ones(total, dtype=np.float32)
    for a, b in merge_spans(spans, bridge=attack + release):
        a, b = max(0, a), min(total, b)
        if a >= b:
            continue
        db[a:b] = depth_db
        if attack > 0:
            lo = a - attack
            ramp = np.linspace(0.0, depth_db, attack, endpoint=False, dtype=np.float32)
            seg = slice(max(0, lo), a)
            db[seg] = np.minimum(db[seg], ramp[max(0, lo) - lo :])
        if release > 0:
            hi = min(total, b + release)
            ramp = np.linspace(depth_db, 0.0, release + 1, dtype=np.float32)[1:]
            db[b:hi] = np.minimum(db[b:hi], ramp[: hi - b])
    gain: NDArray[np.float32] = np.power(np.float32(10), db / np.float32(20)).astype(np.float32)
    return gain


def mix(total: int, layers: Sequence[Layer], settings: MixSettings) -> tuple[Audio, MixReport]:
    """All layers summed into one buffer of exactly `total` samples (no automatic gain)."""
    ch = settings.channels
    out = np.zeros((total, ch), dtype=np.float32)
    fade = settings.samples(settings.edge_fade_ms)
    for layer in layers:
        env = None
        if layer.duck_db is not None and layer.duck_under:
            env = duck_envelope(
                total,
                layer.duck_under,
                layer.duck_db,
                settings.samples(settings.duck_attack_ms),
                settings.samples(settings.duck_release_ms),
            )
        for p in layer.placements:
            audio = _channels(p.audio, ch)
            n = len(audio)
            lo, hi = max(0, p.start), min(total, p.start + n)
            if lo >= hi:
                continue
            gain = edge_fades(n, fade)[lo - p.start : hi - p.start] * np.float32(
                db_to_gain(p.gain_db)
            )
            if env is not None:
                gain = gain * env[lo:hi]
            out[lo:hi] += audio[lo - p.start : hi - p.start] * gain[:, None]
    peak = float(np.abs(out).max()) if total else 0.0
    return out, MixReport(peak=peak, clipped=int(np.count_nonzero(np.abs(out) > 1.0)))


def seam_jumps(audio: Audio, seams: Sequence[int], radius: int) -> list[float]:
    """For each seam, the largest step between adjacent samples within `radius` samples of it
    (any channel). A click is a step far above what the material around it has."""
    out: list[float] = []
    for s in seams:
        lo, hi = max(0, s - radius), min(len(audio), s + radius + 1)
        if hi - lo < 2:
            out.append(0.0)
            continue
        out.append(float(np.abs(np.diff(audio[lo:hi], axis=0)).max()))
    return out


def _channels(audio: Audio, ch: int) -> Audio:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == ch:
        return audio
    if audio.shape[1] == 1:
        return np.repeat(audio, ch, axis=1)
    if ch == 1:
        return audio.mean(axis=1, keepdims=True, dtype=np.float32)
    raise ValueError(f"cannot mix {audio.shape[1]} channels into {ch}")
