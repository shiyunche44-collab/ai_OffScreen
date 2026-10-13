"""Whole-programme mixdown on sample arrays (ARCHITECTURE §7.5 step ②). Pure numpy, no IO.

Narration, the film's own sound and background music are placed by sample position into one
stereo float buffer. Every piece gets a short fade at both ends so cuts do not click, and music
can be ducked under the narration by a gain envelope with configurable attack and release."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Samples = NDArray[np.float32]
"""Shape `(n, 2)`, float32, nominally within [-1, 1]."""

DEFAULT_FADE_MS = 10.0
DEFAULT_ATTACK_MS = 60.0
DEFAULT_RELEASE_MS = 400.0
MUSIC_FADE_OUT_MS = 2000.0
MUSIC_CROSSFADE_MS = 500.0


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def ms_to_samples(ms: float, rate: int) -> int:
    return round(ms * rate / 1000)


@dataclass(frozen=True)
class Placed:
    """A piece of sound starting at sample `start` (may be negative: the head is cut off)."""

    samples: Samples
    start: int
    gain_db: float = 0.0
    fade_ms: float = DEFAULT_FADE_MS


@dataclass(frozen=True)
class MusicPlaced:
    """Background music over `[start, end)`, optionally ducked under the narration."""

    samples: Samples
    start: int
    end: int
    gain_db: float = -22.0
    duck_depth_db: float | None = None
    fade_ms: float = DEFAULT_FADE_MS
    fade_out_ms: float = 0.0
    """A longer fade to silence ending at `end`, so the music does not stop abruptly."""
    crossfade_ms: float = 0.0
    """Overlap between repeats when the track is shorter than `[start, end)`."""


def with_fades(x: Samples, fade_n: int) -> Samples:
    """A copy of `x` with linear fade-in and fade-out of `fade_n` samples (shortened so the two
    never overlap on a short piece)."""
    out = x.astype(np.float32, copy=True)
    n = len(out)
    k = min(max(fade_n, 0), n // 2)
    if k > 0:
        ramp = (np.arange(k, dtype=np.float32) + 1.0) / (k + 1.0)
        out[:k] *= ramp[:, None]
        out[n - k :] *= ramp[::-1, None]
    return out


def fit_music(x: Samples, length: int, *, fade_out_n: int = 0, crossfade_n: int = 0) -> Samples:
    """Exactly `length` samples of music: the track trimmed, or repeated (each repeat blended
    into the previous over `crossfade_n` samples so the seam does not click), with a fade to
    silence over the last `fade_out_n` samples."""
    if length <= 0 or len(x) == 0:
        return np.zeros((max(length, 0), 2), dtype=np.float32)
    out = x[:length].astype(np.float32, copy=True)
    if len(out) < length:
        cf = min(max(crossfade_n, 0), len(x) // 2)
        up = (np.arange(cf, dtype=np.float32) + 1.0) / (cf + 1.0)
        pieces = [out]
        have = len(out)
        while have < length:
            nxt = x.astype(np.float32, copy=True)
            if cf > 0:
                prev = pieces[-1]
                blend = prev[-cf:] * up[::-1, None] + nxt[:cf] * up[:, None]
                pieces[-1] = prev[:-cf]
                nxt = np.concatenate([blend, nxt[cf:]])
                have -= cf
            pieces.append(nxt)
            have += len(nxt)
        out = np.concatenate(pieces)[:length]
    k = min(max(fade_out_n, 0), len(out))
    if k > 0:
        out[len(out) - k :] *= np.linspace(1.0, 0.0, k, dtype=np.float32)[:, None]
    return out


def duck_envelope(
    n: int,
    spans: Sequence[tuple[int, int]],
    *,
    depth_db: float,
    rate: int,
    attack_ms: float = DEFAULT_ATTACK_MS,
    release_ms: float = DEFAULT_RELEASE_MS,
) -> NDArray[np.float32]:
    """Linear gain per sample: 1.0 away from the spans, `depth_db` inside them. The gain falls
    during the `attack` just before a span begins (so the music is already down when the voice
    starts) and recovers during the `release` after it ends; between nearby spans the lower
    of the two ramps wins, so a short pause does not pump the music back up."""
    depth = min(float(depth_db), 0.0)
    env_db = np.zeros(max(n, 0), dtype=np.float64)
    attack = max(ms_to_samples(attack_ms, rate), 1)
    release = max(ms_to_samples(release_ms, rate), 1)
    for a, b in spans:
        if b <= a:
            continue
        lo, hi = max(0, a - attack), min(n, b + release)
        if lo >= hi:
            continue
        xs = np.arange(lo, hi, dtype=np.float64)
        seg = np.interp(xs, [a - attack, a, b, b + release], [0.0, depth, depth, 0.0])
        np.minimum(env_db[lo:hi], seg, out=env_db[lo:hi])
    return (10.0 ** (env_db / 20.0)).astype(np.float32)


def _add(buf: Samples, piece: Samples, start: int) -> None:
    """buf[start:start+len] += piece, clipped to the buffer."""
    lo, hi = max(start, 0), min(start + len(piece), len(buf))
    if lo < hi:
        buf[lo:hi] += piece[lo - start : hi - start]


def mix(
    total: int,
    *,
    rate: int,
    narration: Sequence[Placed] = (),
    source: Sequence[Placed] = (),
    music: Sequence[MusicPlaced] = (),
    attack_ms: float = DEFAULT_ATTACK_MS,
    release_ms: float = DEFAULT_RELEASE_MS,
) -> Samples:
    """The programme's sound: exactly `total` samples of stereo. No automatic gain change and no
    clipping here; loudness is the next step's job."""
    if total <= 0:
        raise ValueError("total must be positive")
    out = np.zeros((total, 2), dtype=np.float32)
    voice_spans: list[tuple[int, int]] = []
    for p in narration:
        piece = with_fades(p.samples, ms_to_samples(p.fade_ms, rate)) * np.float32(
            db_to_gain(p.gain_db)
        )
        _add(out, piece, p.start)
        voice_spans.append((p.start, p.start + len(p.samples)))
    for p in source:
        piece = with_fades(p.samples, ms_to_samples(p.fade_ms, rate)) * np.float32(
            db_to_gain(p.gain_db)
        )
        _add(out, piece, p.start)
    for m in music:
        length = min(m.end, total) - max(m.start, 0)
        if length <= 0:
            continue
        body = fit_music(
            m.samples,
            m.end - m.start,
            fade_out_n=ms_to_samples(m.fade_out_ms, rate),
            crossfade_n=ms_to_samples(m.crossfade_ms, rate),
        )
        piece = with_fades(body, ms_to_samples(m.fade_ms, rate)) * np.float32(db_to_gain(m.gain_db))
        if m.duck_depth_db is not None and voice_spans:
            env = duck_envelope(
                total,
                voice_spans,
                depth_db=m.duck_depth_db,
                rate=rate,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
            lo, hi = max(m.start, 0), min(m.start + len(piece), total)
            if lo < hi:
                piece = piece.copy()
                piece[lo - m.start : hi - m.start] *= env[lo:hi, None]
        _add(out, piece, m.start)
    return out


def adjacent_jump(x: Samples) -> float:
    """The largest sample-to-sample step in the buffer (click detector for tests and reports)."""
    if len(x) < 2:
        return 0.0
    return float(np.abs(np.diff(x, axis=0)).max())
