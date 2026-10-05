"""The real InsightFace model on a real portrait (scikit-image's astronaut photo). Needs the
`insightface` and `onnxruntime` packages and downloads the ~280 MB `buffalo_l` model on first
use."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from offscreen.providers.adapters.insightface_faces import InsightFaceAnalyzer

pytestmark = pytest.mark.heavy


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


def test_a_portrait_gives_one_face_with_a_unit_feature_and_the_same_person_matches(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    data = pytest.importorskip("skimage.data")
    pytest.importorskip("insightface")
    pytest.importorskip("onnxruntime")

    rgb = data.astronaut()
    plain = tmp_path / "astronaut.jpg"
    flipped = tmp_path / "astronaut_flipped.jpg"
    cv2.imwrite(str(plain), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(flipped), cv2.cvtColor(rgb[:, ::-1], cv2.COLOR_RGB2BGR))

    analyzer = InsightFaceAnalyzer("buffalo_l")
    (face,) = analyzer.detect_and_embed(plain)
    assert face.score > 0.7 and len(face.embedding) == 512
    assert math.isclose(sum(v * v for v in face.embedding), 1.0, rel_tol=1e-3)
    x0, y0, x1, y1 = face.bbox
    assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0
    assert 0.3 < (x0 + x1) / 2 < 0.55 and (y1 - y0) < 0.4  # the head sits in the upper middle

    (mirrored,) = analyzer.detect_and_embed(flipped)
    assert cosine(face.embedding, mirrored.embedding) > 0.6  # the same person, mirrored
