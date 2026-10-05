"""Face detection and identity features with InsightFace (SCRFD detector + ArcFace embedder).
The package is imported lazily, so everything else works without it; the model files are
downloaded by InsightFace on first use."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from offscreen.providers.ports import DetectedFace

# Raise `VERSION` when the settings below change: it is part of the cache key.
VERSION = 1
DET_SIZE = (640, 640)


def _unit(v: float) -> float:
    return min(1.0, max(0.0, v))


class InsightFaceAnalyzer:
    def __init__(
        self, model: str = "buffalo_l", *, device: str = "cpu", engine: Any = None
    ) -> None:
        """`engine` is a ready `FaceAnalysis`-like object (tests inject a fake)."""
        self.model = model
        self.device = device
        self._engine = engine

    @property
    def id(self) -> str:
        # the device only changes numerics marginally: moving machines keeps the cache
        return f"insightface/{self.model}@{VERSION}/det{DET_SIZE[0]}"

    def _load(self) -> Any:
        if self._engine is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as e:
                raise RuntimeError(
                    "insightface is not installed; install insightface and onnxruntime "
                    "(onnxruntime-gpu for device: cuda) to detect faces"
                ) from e
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            engine = FaceAnalysis(name=self.model, providers=providers)
            engine.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=DET_SIZE)
            self._engine = engine
        return self._engine

    def detect_and_embed(self, image: Path) -> list[DetectedFace]:
        import cv2

        # imdecode + fromfile, so paths with non-ASCII characters work on every platform
        frame = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"cannot read the image {image}")
        height, width = frame.shape[:2]
        found: list[DetectedFace] = []
        for face in self._load().get(frame):
            x0, y0, x1, y1 = (float(v) for v in face.bbox)
            box = (_unit(x0 / width), _unit(y0 / height), _unit(x1 / width), _unit(y1 / height))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            found.append(
                DetectedFace(
                    bbox=box,
                    score=_unit(float(face.det_score)),
                    embedding=tuple(float(v) for v in face.normed_embedding),
                )
            )
        return found
