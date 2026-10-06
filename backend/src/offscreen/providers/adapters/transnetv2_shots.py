"""ShotDetector backed by TransNetV2, a neural network trained to find cuts and gradual
transitions (dissolves, fades) that colour-difference detectors miss. Uses the PyTorch port in
the `transnetv2-pytorch` package, which ships the weights (install it to use this detector:
`uv pip install transnetv2-pytorch`). Frames come from ffmpeg (media/), at the 48x27 the model
wants, so a film costs about 4 KB per frame in memory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from offscreen.algo.transitions import cut_frames, spans_from_cuts
from offscreen.domain.common import TimeRange
from offscreen.media.ffmpeg import FFmpegCanceled, read_raw_video
from offscreen.media.probe import probe
from offscreen.providers.ports import DetectionCanceled

# Raise `VERSION` when the settings below change: it is part of the cache key.
VERSION = 1
INPUT_SIZE = (48, 27)  # width, height the network is trained on
DEFAULT_THRESHOLD = 0.5
READ_SHARE = 0.4
"""Share of the progress bar taken by decoding (the rest is the network)."""


class TransNetV2Shots:
    """Cuts and gradual transitions; each transition becomes one boundary (right after it)."""

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        device: str = "cpu",
        model: Any = None,
    ) -> None:
        """`model` is a ready TransNetV2-like object (tests inject a fake)."""
        self.threshold = threshold
        self.device = device
        self._model = model

    @property
    def id(self) -> str:
        return f"transnetv2/pytorch@{VERSION}:t{self.threshold}"

    def _load(self) -> Any:
        if self._model is None:
            try:
                from transnetv2_pytorch import TransNetV2
            except ImportError as e:
                raise RuntimeError(
                    "transnetv2-pytorch is not installed; install it to use this detector "
                    "(uv pip install transnetv2-pytorch)"
                ) from e
            self._model = TransNetV2(device=self.device)
        return self._model

    def detect(
        self,
        video: Path,
        *,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[TimeRange]:
        info = probe(video)
        model = self._load()
        w, h = INPUT_SIZE
        try:
            raw = read_raw_video(
                str(video),
                width=w,
                height=h,
                duration_ms=info.duration_ms,
                on_progress=(lambda f: on_progress(READ_SHARE * f)) if on_progress else None,
                should_cancel=should_cancel,
            )
        except FFmpegCanceled as e:
            raise DetectionCanceled("shot detection canceled") from e
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w, 3)
        if len(frames) == 0:
            return []
        if should_cancel is not None and should_cancel():
            raise DetectionCanceled("shot detection canceled")
        scores = self._scores(model, frames, on_progress, should_cancel)
        fps = info.video.fps
        spans = spans_from_cuts(cut_frames(scores, self.threshold), len(frames), fps.num, fps.den)
        return [TimeRange(start_ms=a, end_ms=b) for a, b in spans]

    def _scores(
        self,
        model: Any,
        frames: np.ndarray,
        on_progress: Callable[[float], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> list[float]:
        """Per-frame transition scores, window by window so cancel and progress work."""
        import torch

        # the library slides 100-frame windows by 50; ask it for the whole film in pieces
        # that overlap enough (50 frames each side) to give the same answer as one call
        piece, margin = 3000, 50
        n = len(frames)
        scores = np.zeros(n, dtype=np.float32)
        start = 0
        while start < n:
            if should_cancel is not None and should_cancel():
                raise DetectionCanceled("shot detection canceled")
            lo, hi = max(0, start - margin), min(n, start + piece + margin)
            chunk = torch.from_numpy(np.array(frames[lo:hi]))  # a writable copy.to(model.device)
            single, _ = model.predict_frames(chunk, quiet=True)
            part = single.detach().cpu().numpy().reshape(-1)
            take_hi = min(n, start + piece)
            scores[start:take_hi] = part[start - lo : take_hi - lo]
            start = take_hi
            if on_progress:
                on_progress(READ_SHARE + (1 - READ_SHARE) * start / n)
        return [float(s) for s in scores]
