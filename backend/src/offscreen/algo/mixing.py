"""Mixdown of the whole programme's sound onto one sample grid (ARCHITECTURE §7.5). Pure numpy.

Everything is placed by sample position in one buffer (stereo, float32), so there are no
per-segment encodes and no drift. Narration is laid down as is; the original sound and the
BGM can be ducked under it by a smooth gain envelope; every piece of source sound gets a short
fade at both ends so cuts between pieces do not click. Nothing is clipped here: loudness and
peak control come after the mix (M6-07)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

RATE = 48_000
Audio = NDArray[np.float32]
"""Shape `(n_samples, 2)`."""

FADE_MS_DEFAULT = 15
"""Within the 10-20 ms the design asks for."""
ATTACK_MS_DEFAULT = 40
RELEASE_MS_DEFAULT = 300


def ms_to_samples(ms: float, rate: int = RATE) -> int:
    return round(ms * rate / 1000)


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


@dataclass(frozen=True)
class Duck:
    depth_db: float = -8.0
    """Gain change while narration plays (negative: quieter)."""
    attack_ms: float = ATTACK_MS_DEFAULT
    """Ramp down that ends as the narration starts."""
    release_ms: float = RELEASE_MS_DEFAULT
    """Ramp back up after the narration ends."""


@dataclass(frozen=True)
class Piece:
    """Sound placed on the programme: `samples` start at sample `start` of the output."""

    start: int
    samples: Audio
    gain_db: float = 0.0
    fade_ms: float = 0.0
    """Fade in and out on the piece's own ends (0: none)."""
    duck: Duck | None = None
    """Lower this piece while narration plays."""

    @property
    def end(self) -> int:
        return self.start + len(self.samples)


def fade_edges(x: Audio, n: int) -> Audio:
    """Linear fade in and out over `n` samples each (never more than half the piece)."""
    n = min(n, len(x) // 2)
    if n <= 0:
        return x
    out = x.copy()
    ramp = (np.arange(n, dtype=np.float32) + 1) / (n + 1)
    out[:n] *= ramp[:, None]
    out[-n:] *= ramp[::-1][:, None]
    return out


def loop_to(x: Audio, n: int) -> Audio:
    """Repeat `x` end to end until it is `n` samples long (cut at `n`)."""
    if n <= 0 or len(x) == 0:
        return x[:0]
    reps = -(-n // len(x))
    return np.tile(x, (reps, 1))[:n]


def merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and join spans that touch or overlap; empty spans are dropped."""
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(s for s in spans if s[0] < s[1]):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def duck_envelope(
    total: int, spans: Sequence[tuple[int, int]], duck: Duck, rate: int = RATE
) -> NDArray[np.float32]:
    """Linear gain per sample: 1.0 away from the spans, `depth_db` inside them, with a ramp
    that reaches the depth exactly when a span starts and leaves it after the span ends.
    Where ramps of neighbouring spans meet, the lower gain wins."""
    env = np.ones(total, dtype=np.float32)
    low = np.float32(db_to_gain(duck.depth_db))
    attack = ms_to_samples(duck.attack_ms, rate)
    release = ms_to_samples(duck.release_ms, rate)
    for lo, hi in merge_spans(spans):
        lo, hi = max(0, lo), min(total, hi)
        if lo >= hi:
            continue
        a0, r1 = max(0, lo - attack), min(total, hi + release)
        down = np.linspace(1.0, low, lo - a0 + 1, dtype=np.float32)[:-1] if lo > a0 else None
        up = np.linspace(low, 1.0, r1 - hi + 1, dtype=np.float32)[1:] if r1 > hi else None
        seg = np.concatenate(
            [p for p in (down, np.full(hi - lo, low, dtype=np.float32), up) if p is not None]
        )
        np.minimum(env[a0:r1], seg, out=env[a0:r1])
    return env


def _add(out: Audio, piece: Piece, env: NDArray[np.float32] | None, rate: int) -> None:
    lo, hi = max(0, piece.start), min(len(out), piece.end)
    if lo >= hi:
        return
    x = piece.samples
    if piece.fade_ms:
        x = fade_edges(x, ms_to_samples(piece.fade_ms, rate))
    x = x[lo - piece.start : hi - piece.start] * np.float32(db_to_gain(piece.gain_db))
    if piece.duck is not None and env is not None:
        x = x * env[lo:hi, None]
    out[lo:hi] += x


def mix(
    total: int,
    narration: Sequence[Piece],
    others: Sequence[Piece] = (),
    rate: int = RATE,
) -> Audio:
    """The programme's sound, `total` samples long: `narration` pieces as they are, `others`
    (original sound, BGM) beneath, ducked where a piece asks for it.

    Pieces are cut at the edges of the buffer. Narration pieces ignore `duck`."""
    out = np.zeros((total, 2), dtype=np.float32)
    spans = [(p.start, p.end) for p in narration]
    envs: dict[Duck, NDArray[np.float32]] = {}
    for p in others:
        if p.duck is not None and p.duck not in envs:
            envs[p.duck] = duck_envelope(total, spans, p.duck, rate)
    for p in others:
        _add(out, p, envs.get(p.duck) if p.duck is not None else None, rate)
    for p in narration:
        _add(out, Piece(p.start, p.samples, p.gain_db, p.fade_ms), None, rate)
    return out


def peak_db(x: Audio) -> float:
    """Highest absolute sample, in dB full scale (-inf for silence)."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return float("-inf") if peak == 0.0 else 20.0 * float(np.log10(peak))


def max_step(x: Audio) -> float:
    """Largest jump between neighbouring samples on any channel: a click detector."""
    return float(np.max(np.abs(np.diff(x, axis=0)))) if len(x) > 1 else 0.0
