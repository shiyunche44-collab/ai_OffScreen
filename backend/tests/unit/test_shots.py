from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from offscreen.domain.common import TimeRange
from offscreen.domain.index import Shots
from offscreen.engine import Artifact, ArtifactStore, Engine, StageCanceled
from offscreen.media.probe import probe
from offscreen.providers.adapters.scenedetect_adapter import SceneDetectShots
from offscreen.providers.ports import DetectionCanceled
from offscreen.stages.analysis.ingest import ingest
from offscreen.stages.analysis.keyframes import KeyframesStage
from offscreen.stages.analysis.proxy import ProxyError, ProxyStage
from offscreen.stages.analysis.shots import SHOTS_FILE, ShotsStage
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo


@pytest.fixture(scope="module")
def cut_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three 2 s solid-colour segments: hard cuts at 2000 and 4000 ms."""
    p = tmp_path_factory.mktemp("cuts") / "cuts.mp4"
    parts = "".join(
        f"color=c={c}:s=320x180:r=24:d=2[v{i}];" for i, c in enumerate(("red", "blue", "green"))
    )
    graph = parts + "[v0][v1][v2]concat=n=3:v=1:a=0[v]"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-filter_complex", graph, "-map", "[v]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
        check=True,
    )  # fmt: skip
    return p


def jpeg_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    w, h = out.strip().split(",")
    return int(w), int(h)


class FakeDetector:
    id = "fake@1"

    def __init__(self, shots: list[tuple[int, int]]) -> None:
        self.shots = shots
        self.calls = 0

    def detect(
        self,
        video: Path,
        *,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[TimeRange]:
        self.calls += 1
        return [TimeRange(start_ms=a, end_ms=b) for a, b in self.shots]


def build(tmp_path: Path, detector: object, **engine_kw: object) -> tuple[AssetRepo, Engine]:
    db = Database(tmp_path / "db" / "o.db")
    repo = AssetRepo(db)
    stages = [ProxyStage(repo), ShotsStage(repo, detector), KeyframesStage()]  # type: ignore[arg-type]
    return repo, Engine(ArtifactStore(tmp_path / "artifacts"), stages, **engine_kw)  # type: ignore[arg-type]


def test_pyscenedetect_finds_the_hard_cuts(cut_clip: Path) -> None:
    found = SceneDetectShots().detect(cut_clip)
    assert [(s.start_ms, s.end_ms) for s in found] == [(0, 2000), (2000, 4000), (4000, 6000)]


def test_detector_can_be_canceled(cut_clip: Path) -> None:
    with pytest.raises(DetectionCanceled):
        SceneDetectShots().detect(cut_clip, should_cancel=lambda: True)


def test_shots_stage_writes_valid_shots_json(cut_clip: Path, tmp_path: Path) -> None:
    repo, engine = build(tmp_path, SceneDetectShots())
    asset = ingest(cut_clip, repo).asset
    art = engine.ensure("analysis.shots", {"asset_id": asset.id})

    doc = art.read_model(SHOTS_FILE, Shots)
    assert doc.asset_id == asset.id
    assert [(s.id, s.start_ms, s.end_ms) for s in doc.shots] == [
        ("sh_0001", 0, 2000),
        ("sh_0002", 2000, 4000),
        ("sh_0003", 4000, 6000),
    ]
    assert art.meta == {"shots": 3, "raw_cuts": 2}


def test_shots_are_postprocessed_to_cover_the_video(cut_clip: Path, tmp_path: Path) -> None:
    # Two fragments at the start fold into the 6 s shot after them (6 s is under the 8 s cap).
    fake = FakeDetector([(0, 150), (150, 400), (400, 6000)])
    repo, engine = build(tmp_path, fake)
    asset = ingest(cut_clip, repo).asset
    doc = engine.ensure("analysis.shots", {"asset_id": asset.id}).read_model(SHOTS_FILE, Shots)
    end = probe(engine.ensure("analysis.proxy", {"asset_id": asset.id}).path("proxy_540p.mp4"))
    assert [(s.start_ms, s.end_ms) for s in doc.shots] == [(0, end.duration_ms)]


def test_shots_cache_depends_on_detector_and_hits_otherwise(cut_clip: Path, tmp_path: Path) -> None:
    fake = FakeDetector([(0, 2000), (2000, 6000)])
    repo, engine = build(tmp_path, fake)
    asset = ingest(cut_clip, repo).asset
    first = engine.ensure("analysis.shots", {"asset_id": asset.id})
    again = engine.ensure("analysis.shots", {"asset_id": asset.id})
    assert again.dir == first.dir and fake.calls == 1

    fake.id = "fake@2"
    changed = engine.ensure("analysis.shots", {"asset_id": asset.id})
    assert changed.dir != first.dir and fake.calls == 2


def test_keyframes_one_middle_thumbnail_per_shot(cut_clip: Path, tmp_path: Path) -> None:
    repo, engine = build(tmp_path, SceneDetectShots())
    asset = ingest(cut_clip, repo).asset
    art: Artifact = engine.ensure("analysis.keyframes", {"asset_id": asset.id})

    doc = art.read_model(SHOTS_FILE, Shots)
    assert [s.keyframes for s in doc.shots] == [[f"kf/sh_000{i}.jpg"] for i in (1, 2, 3)]
    for shot in doc.shots:
        assert jpeg_size(art.path(shot.keyframes[0])) == (320, 180)  # not upscaled
    assert art.meta == {"keyframes": 3}


def test_keyframe_shows_the_middle_of_its_shot(cut_clip: Path, tmp_path: Path) -> None:
    repo, engine = build(tmp_path, SceneDetectShots())
    asset = ingest(cut_clip, repo).asset
    art = engine.ensure("analysis.keyframes", {"asset_id": asset.id})
    expected = [(255, 0, 0), (0, 0, 255), (0, 128, 0)]  # red, blue, green (approx.)
    for i, (r, g, b) in enumerate(expected, 1):
        px = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(art.path(f"kf/sh_000{i}.jpg")),
             "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True,
        ).stdout  # fmt: skip
        assert all(abs(a - e) < 60 for a, e in zip(px, (r, g, b), strict=True)), (i, list(px))


def test_unknown_asset_is_an_error(tmp_path: Path) -> None:
    _, engine = build(tmp_path, FakeDetector([]))
    with pytest.raises(ProxyError, match="unknown asset"):  # the proxy is resolved first
        engine.ensure("analysis.shots", {"asset_id": "ast_nope"})


def test_cancel_during_keyframes_leaves_no_artifact(cut_clip: Path, tmp_path: Path) -> None:
    repo, warm = build(tmp_path, SceneDetectShots())
    asset = ingest(cut_clip, repo).asset
    warm.ensure("analysis.shots", {"asset_id": asset.id})  # upstream is cached

    canceled = {"on": False}

    def progress(stage: str, _frac: float, _msg: str) -> None:
        canceled["on"] = canceled["on"] or stage == "analysis.keyframes"

    _, engine = build(
        tmp_path, SceneDetectShots(), is_canceled=lambda: canceled["on"], progress=progress
    )
    with pytest.raises(StageCanceled):
        engine.ensure("analysis.keyframes", {"asset_id": asset.id})
    assert list((tmp_path / "artifacts" / "analysis.keyframes").iterdir()) == []
