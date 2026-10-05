"""The faces stage, its pure helpers and the InsightFace adapter (with a stand-in engine)."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.faces import box_area, unit_vector
from offscreen.domain.index import Faces, Shot, Shots
from offscreen.engine import (
    ArtifactStore,
    Engine,
    Scope,
    Stage,
    StageCanceled,
    StageContext,
    StageOutput,
)
from offscreen.providers.adapters.fake import FakeFaceAnalyzer
from offscreen.providers.adapters.insightface_faces import InsightFaceAnalyzer
from offscreen.providers.ports import DetectedFace
from offscreen.stages.analysis.faces import (
    EMBEDDINGS_FILE,
    FACES_FILE,
    MIN_AREA_RATIO,
    FacesError,
    FacesStage,
)
from offscreen.stages.analysis.shots import SHOTS_FILE
from offscreen.store.files import write_model

ASSET = "ast_f1"
SCOPE = {"asset_id": ASSET}


class StubKeyframes(Stage):
    """`n` shots with three (empty) keyframe files each."""

    name = "analysis.keyframes"
    version = 1
    lane = "cpu"

    def __init__(self, n: int = 3) -> None:
        self.n = n

    def params(self, scope: Scope) -> dict[str, Any]:
        return {"n": self.n}

    def run(self, ctx: StageContext) -> StageOutput:
        (ctx.out_dir / "kf").mkdir()
        shots = []
        for i in range(self.n):
            frames = [f"kf/sh_{i}_{s}.jpg" for s in "abc"]
            for rel in frames:
                (ctx.out_dir / rel).write_bytes(b"jpg")
            shots.append(
                Shot(id=f"sh_{i}", start_ms=i * 1000, end_ms=(i + 1) * 1000, keyframes=frames)
            )
        write_model(ctx.out_dir / SHOTS_FILE, Shots(asset_id=ASSET, shots=shots))
        return StageOutput()


def face(
    x0: float = 0.2,
    y0: float = 0.1,
    x1: float = 0.5,
    y1: float = 0.6,
    *,
    score: float = 0.9,
    vec: tuple[float, ...] = (3.0, 4.0),
) -> DetectedFace:
    return DetectedFace(bbox=(x0, y0, x1, y1), score=score, embedding=vec)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def run(store: ArtifactStore, analyzer: FakeFaceAnalyzer | None, n: int = 3) -> Any:
    return Engine(store, [StubKeyframes(n), FacesStage(analyzer)]).ensure("analysis.faces", SCOPE)


# --- helpers ------------------------------------------------------------------------------


def test_box_area_and_unit_vector() -> None:
    assert box_area((0.1, 0.1, 0.5, 0.6)) == pytest.approx(0.2)
    assert box_area((0.5, 0.5, 0.2, 0.9)) == 0.0  # inverted: nothing
    assert unit_vector((3.0, 4.0)) == pytest.approx([0.6, 0.8])
    assert unit_vector((0.0, 0.0)) is None
    assert unit_vector((math.nan, 1.0)) is None
    assert unit_vector((math.inf, 1.0)) is None


@given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=16))
def test_unit_vectors_have_length_one(v: list[float]) -> None:
    out = unit_vector(v)
    if out is not None:
        assert math.isclose(sum(x * x for x in out), 1.0, rel_tol=1e-9)


# --- the stage ----------------------------------------------------------------------------


def test_every_keyframe_is_analysed_and_rows_follow_document_order(store: ArtifactStore) -> None:
    def faces_in(image: Path) -> list[DetectedFace]:
        if image.name == "sh_1_b.jpg":  # two faces; the larger one comes first
            return [
                face(0.1, 0.1, 0.2, 0.3, vec=(0.0, 2.0)),
                face(0.3, 0.1, 0.8, 0.9, vec=(5.0, 0.0)),
            ]
        return []

    analyzer = FakeFaceAnalyzer(faces_in)
    art = run(store, analyzer)
    assert len(analyzer.images) == 9  # 3 shots x 3 keyframes
    doc = art.read_model(FACES_FILE, Faces)
    assert [s.shot_id for s in doc.shots] == ["sh_0", "sh_1", "sh_2"]
    assert [len(s.faces) for s in doc.shots] == [0, 2, 0]
    big, small = doc.shots[1].faces
    assert big.area_ratio == pytest.approx(0.4) and small.area_ratio == pytest.approx(0.02)
    assert (big.frame, big.embedding, small.embedding) == (1, 0, 1)
    assert big.score == 0.9 and big.character_id is None

    matrix = np.load(art.path(EMBEDDINGS_FILE))
    assert matrix.dtype == np.float32 and matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix, [[1.0, 0.0], [0.0, 1.0]])  # unit length, by embedding row
    assert art.meta == {"shots": 3, "shots_with_faces": 1, "faces": 2}


def test_unsure_tiny_and_featureless_faces_are_dropped(store: ArtifactStore) -> None:
    side = math.sqrt(MIN_AREA_RATIO) * 0.9
    analyzer = FakeFaceAnalyzer(
        lambda p: (
            [
                face(score=0.3),  # not sure it is a face
                face(0.1, 0.1, 0.1 + side, 0.1 + side),  # too small
                face(vec=(0.0, 0.0)),  # no usable feature
                face(vec=(1.0, 0.0)),  # kept
            ]
            if p.name == "sh_0_a.jpg"
            else []
        )
    )
    art = run(store, analyzer, n=1)
    assert art.meta["faces"] == 1
    assert len(np.load(art.path(EMBEDDINGS_FILE))) == 1


def test_no_faces_at_all_still_gives_a_valid_artifact(store: ArtifactStore) -> None:
    art = run(store, FakeFaceAnalyzer())
    assert [len(s.faces) for s in art.read_model(FACES_FILE, Faces).shots] == [0, 0, 0]
    assert np.load(art.path(EMBEDDINGS_FILE)).shape == (0, 0)


def test_features_of_different_lengths_are_an_error(store: ArtifactStore) -> None:
    analyzer = FakeFaceAnalyzer(
        lambda p: (
            [face(vec=(1.0, 0.0))] if p.name.endswith("a.jpg") else [face(vec=(1.0, 0.0, 0.0))]
        )
    )
    with pytest.raises(FacesError, match="different lengths"):
        run(store, analyzer, n=1)


def test_without_an_analyzer_the_stage_says_so(store: ArtifactStore) -> None:
    with pytest.raises(FacesError, match="no face analyzer"):
        run(store, None)


def test_cached_until_the_analyzer_changes(store: ArtifactStore) -> None:
    first = FakeFaceAnalyzer()
    run(store, first)
    again = FakeFaceAnalyzer()
    run(store, again)
    assert again.images == []  # served from the cache

    other = FakeFaceAnalyzer()
    other.id = "fake-faces@2"  # type: ignore[misc]
    run(store, other)
    assert len(other.images) == 9


def test_cancellation_stops_between_shots(store: ArtifactStore) -> None:
    analyzer = FakeFaceAnalyzer()
    stages = [StubKeyframes(3), FacesStage(analyzer)]
    seen = {"n": 0}

    def canceled() -> bool:
        seen["n"] += 1
        return seen["n"] > 2  # the keyframes stub asks once, the stage once per shot

    with pytest.raises(StageCanceled):
        Engine(store, stages, is_canceled=canceled).ensure("analysis.faces", SCOPE)
    assert len(analyzer.images) < 9


# --- the InsightFace adapter --------------------------------------------------------------


class StubEngine:
    def __init__(self, faces: list[Any]) -> None:
        self.faces = faces
        self.frames: list[Any] = []

    def get(self, frame: Any) -> list[Any]:
        self.frames.append(frame)
        return self.faces


def jpeg(path: Path, width: int, height: int) -> None:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.full((height, width, 3), 128, dtype=np.uint8))
    assert ok
    buf.tofile(path)


def test_adapter_normalises_boxes_and_clips_them_to_the_image(tmp_path: Path) -> None:
    image = tmp_path / "图 1.jpg"  # a non-ASCII name must work
    jpeg(image, 200, 100)
    engine = StubEngine(
        [
            SimpleNamespace(bbox=np.array([20.0, 10.0, 120.0, 90.0]), det_score=0.97,
                            normed_embedding=np.array([0.6, 0.8], dtype=np.float32)),
            SimpleNamespace(bbox=np.array([-5.0, 50.0, 250.0, 130.0]), det_score=1.2,
                            normed_embedding=np.array([1.0, 0.0], dtype=np.float32)),
            SimpleNamespace(bbox=np.array([300.0, 10.0, 320.0, 30.0]), det_score=0.9,
                            normed_embedding=np.array([1.0, 0.0], dtype=np.float32)),  # outside
        ]
    )  # fmt: skip
    found = InsightFaceAnalyzer(engine=engine).detect_and_embed(image)
    assert engine.frames[0].shape == (100, 200, 3)
    assert len(found) == 2
    assert found[0].bbox == pytest.approx((0.1, 0.1, 0.6, 0.9)) and found[0].score == 0.97
    assert found[0].embedding == pytest.approx((0.6, 0.8))
    assert found[1].bbox == (0.0, 0.5, 1.0, 1.0) and found[1].score == 1.0


def test_adapter_reports_an_unreadable_image_and_a_missing_package(tmp_path: Path) -> None:
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(RuntimeError, match="cannot read the image"):
        InsightFaceAnalyzer(engine=StubEngine([])).detect_and_embed(bad)

    good = tmp_path / "y.jpg"
    jpeg(good, 10, 10)
    try:
        import insightface  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="insightface is not installed"):
            InsightFaceAnalyzer().detect_and_embed(good)


def test_adapter_id_names_model_and_settings() -> None:
    assert InsightFaceAnalyzer("buffalo_l").id == "insightface/buffalo_l@1/det640"
    assert InsightFaceAnalyzer("buffalo_l", device="cuda").id == InsightFaceAnalyzer("buffalo_l").id


# --- the characters stage on degenerate input -----------------------------------------------


def test_a_film_without_faces_has_no_characters(store: ArtifactStore) -> None:
    from offscreen.domain.index import Cast, Characters
    from offscreen.stages.analysis.characters import (
        CAST_FILE,
        CENTROIDS_FILE,
        CHARACTERS_FILE,
        CharactersStage,
    )

    engine = Engine(store, [StubKeyframes(2), FacesStage(FakeFaceAnalyzer()), CharactersStage()])
    art = engine.ensure("analysis.characters", SCOPE)
    assert art.read_model(CHARACTERS_FILE, Characters).characters == []
    cast = art.read_model(CAST_FILE, Cast)
    assert [(s.shot_id, s.characters) for s in cast.shots] == [("sh_0", []), ("sh_1", [])]
    assert np.load(art.path(CENTROIDS_FILE)).shape == (0, 0)
    assert art.meta == {"faces": 0, "characters": 0, "unassigned_faces": 0}


def test_faces_and_embeddings_that_disagree_are_an_error(store: ArtifactStore) -> None:
    from offscreen.stages.analysis.characters import CharactersError, CharactersStage

    class Lying(FacesStage):
        def run(self, ctx: StageContext) -> StageOutput:
            out = super().run(ctx)
            np.save(ctx.out_dir / EMBEDDINGS_FILE, np.zeros((5, 2), dtype=np.float32))
            return out

    analyzer = FakeFaceAnalyzer(lambda p: [face()] if p.name == "sh_0_a.jpg" else [])
    with pytest.raises(CharactersError, match=r"lists 1 faces but embeddings\.npy has 5 rows"):
        Engine(store, [StubKeyframes(1), Lying(analyzer), CharactersStage()]).ensure(
            "analysis.characters", SCOPE
        )
