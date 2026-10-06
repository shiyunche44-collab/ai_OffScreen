"""TransNetV2 shot detection: score post-processing, the raw frame reader, the adapter (with a
stand-in network)."""

from __future__ import annotations

import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from offscreen.algo.transitions import cut_frames, spans_from_cuts
from offscreen.media.ffmpeg import FFmpegCanceled, FFmpegError, read_raw_video
from offscreen.providers.adapters.transnetv2_shots import TransNetV2Shots
from offscreen.providers.ports import DetectionCanceled

# --- scores -> cuts -> shots ----------------------------------------------------------------


def test_a_single_high_score_cuts_right_after_that_frame() -> None:
    assert cut_frames([0.0, 0.1, 0.9, 0.0, 0.0]) == [3]
    assert cut_frames([0.0] * 5) == []
    assert cut_frames([]) == []


def test_a_run_of_high_scores_is_one_transition_ending_the_shot_after_it() -> None:
    assert cut_frames([0.0, 0.8, 0.9, 0.7, 0.0, 0.0, 0.6, 0.0]) == [4, 7]


def test_a_transition_reaching_the_last_frame_cuts_nothing() -> None:
    assert cut_frames([0.0, 0.0, 0.9]) == []
    assert cut_frames([0.9, 0.0]) == [1]  # a score on the first frame still ends shot one


def test_the_threshold_is_exclusive() -> None:
    assert cut_frames([0.0, 0.5, 0.0], 0.5) == []
    assert cut_frames([0.0, 0.51, 0.0], 0.5) == [2]
    assert cut_frames([0.0, 0.3, 0.0], 0.2) == [2]


@given(st.lists(st.floats(0, 1), max_size=200))
def test_cuts_are_ascending_and_inside_the_film(scores: list[float]) -> None:
    cuts = cut_frames(scores)
    assert cuts == sorted(set(cuts)) and all(1 <= c < len(scores) for c in cuts)


def test_spans_follow_the_frame_rate_and_cover_the_film() -> None:
    spans = spans_from_cuts([24, 48], 72, 24, 1)
    assert spans == [(0, 1000), (1000, 2000), (2000, 3000)]
    ntsc = spans_from_cuts([24], 48, 24000, 1001)
    assert ntsc == [(0, 1001), (1001, 2002)]
    assert spans_from_cuts([], 10, 24, 1) == [(0, 417)]
    # cuts outside the film, at 0, and repeats are ignored
    assert spans_from_cuts([0, 5, 5, 99], 10, 10, 1) == [(0, 500), (500, 1000)]


@given(st.lists(st.integers(-5, 400), max_size=30), st.integers(1, 300))
def test_spans_are_a_partition(cuts: list[int], frames: int) -> None:
    spans = spans_from_cuts(cuts, frames, 24, 1)
    assert spans[0][0] == 0 and spans[-1][1] == round(frames * 1000 / 24)
    assert all(a < b for a, b in spans)
    assert all(x[1] == y[0] for x, y in pairwise(spans))


# --- the raw frame reader ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("raw") / "c.mp4"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=320x180:rate=24:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p),
        ],
        check=True,
    )  # fmt: skip
    return p


def test_every_frame_comes_back_at_the_requested_size(clip: Path) -> None:
    seen: list[float] = []
    data = read_raw_video(str(clip), width=48, height=27, duration_ms=3000, on_progress=seen.append)
    assert len(data) == 72 * 48 * 27 * 3  # 3 s at 24 fps: no frame dropped or repeated
    assert seen and all(0.0 <= f <= 1.0 for f in seen)
    frames = np.frombuffer(data, dtype=np.uint8).reshape(-1, 27, 48, 3)
    assert not np.array_equal(frames[0], frames[-1])  # the test pattern moves


def test_a_small_chunk_size_gives_the_same_bytes(clip: Path) -> None:
    whole = read_raw_video(str(clip), width=32, height=18)
    pieces = read_raw_video(str(clip), width=32, height=18, chunk_frames=5)
    assert whole == pieces


def test_the_reader_can_be_canceled_and_reports_ffmpeg_failures(clip: Path, tmp_path: Path) -> None:
    with pytest.raises(FFmpegCanceled):
        read_raw_video(str(clip), width=48, height=27, should_cancel=lambda: True)
    bad = tmp_path / "nope.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(FFmpegError, match="exit"):
        read_raw_video(str(bad), width=48, height=27)


# --- the adapter, with a stand-in network -----------------------------------------------------


class FakeNet:
    """Scores 0.9 at the frames listed, else 0; records the pieces it was asked about."""

    device = "cpu"

    def __init__(self, hot: set[int], total: int) -> None:
        self.hot, self.total = hot, total
        self.calls: list[int] = []

    def predict_frames(self, frames: Any, quiet: bool = False) -> tuple[Any, Any]:
        import torch

        self.calls.append(len(frames))
        # the chunk's first frame is identified by its content: the stand-in video encodes the
        # frame number in the first pixel's red channel (mod 256) - so use a lookup instead
        offset = int(frames[0, 0, 0, 1]) * 256 + int(frames[0, 0, 0, 2])
        scores = torch.tensor([0.9 if offset + i in self.hot else 0.0 for i in range(len(frames))])
        return scores.reshape(-1, 1)[:, 0], scores


def numbered_video(monkeypatch: pytest.MonkeyPatch, total: int) -> None:
    """Make the raw reader return `total` tiny frames whose green / blue pixels hold the index."""
    frames = np.zeros((total, 27, 48, 3), dtype=np.uint8)
    for i in range(total):
        frames[i, 0, 0, 1], frames[i, 0, 0, 2] = divmod(i, 256)
    monkeypatch.setattr(
        "offscreen.providers.adapters.transnetv2_shots.read_raw_video",
        lambda *a, **k: frames.tobytes(),
    )


def fake_probe(monkeypatch: pytest.MonkeyPatch, fps_num: int = 24, fps_den: int = 1) -> None:
    from types import SimpleNamespace

    from offscreen.domain.common import Rational

    info = SimpleNamespace(
        duration_ms=0, video=SimpleNamespace(fps=Rational(num=fps_num, den=fps_den))
    )
    monkeypatch.setattr("offscreen.providers.adapters.transnetv2_shots.probe", lambda _p: info)


def test_the_adapter_turns_scores_into_consecutive_shots(monkeypatch: pytest.MonkeyPatch) -> None:
    numbered_video(monkeypatch, 240)
    fake_probe(monkeypatch)
    net = FakeNet({95, 190, 191, 192}, 240)
    spans = TransNetV2Shots(model=net).detect(Path("x.mp4"))
    # a one-frame score at 95 cuts at 96; the run 190-192 ends the shot after it, at 193
    assert [(s.start_ms, s.end_ms) for s in spans] == [(0, 4000), (4000, 8042), (8042, 10000)]
    assert TransNetV2Shots(model=net).id == "transnetv2/pytorch@1:t0.5"
    assert TransNetV2Shots(threshold=0.3, model=net).id.endswith(":t0.3")


def test_long_films_are_scored_in_overlapping_pieces_with_the_same_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total = 7000
    hot = {10, 2999, 3000, 3049, 6500}
    numbered_video(monkeypatch, total)
    fake_probe(monkeypatch)
    net = FakeNet(hot, total)
    spans = TransNetV2Shots(model=net).detect(Path("x.mp4"))
    assert len(net.calls) == 3 and max(net.calls) <= 3100  # three pieces, margins included
    expected = cut_frames([0.9 if i in hot else 0.0 for i in range(total)])
    assert len(spans) == len(expected) + 1
    assert [s.start_ms for s in spans][1:] == [round(c * 1000 / 24) for c in expected]


def test_progress_is_reported_and_cancellation_stops_between_pieces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numbered_video(monkeypatch, 7000)
    fake_probe(monkeypatch)
    seen: list[float] = []
    TransNetV2Shots(model=FakeNet(set(), 7000)).detect(Path("x.mp4"), on_progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)

    state = {"n": 0}

    def cancel() -> bool:
        state["n"] += 1
        return state["n"] > 2

    net = FakeNet(set(), 7000)
    with pytest.raises(DetectionCanceled):
        TransNetV2Shots(model=net).detect(Path("x.mp4"), should_cancel=cancel)
    assert len(net.calls) < 3


def test_a_missing_package_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_probe(monkeypatch)
    try:
        import transnetv2_pytorch  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="transnetv2-pytorch is not installed"):
            TransNetV2Shots().detect(Path("x.mp4"))


def test_the_configuration_picks_the_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from offscreen.config import AppConfig
    from offscreen.providers.adapters.fake import FakeTTS
    from offscreen.providers.adapters.scenedetect_adapter import SceneDetectShots
    from offscreen.services.pipeline import build_providers
    from offscreen.store.db import Database

    monkeypatch.setattr("offscreen.services.pipeline.build_tts", lambda _cfg: FakeTTS())
    base = {
        "data_dir": str(tmp_path),
        "asr": {"provider": "faster_whisper"},
        "tts": {"provider": "edge_tts"},
    }
    db = Database(tmp_path / "x.db")
    try:
        default = build_providers(AppConfig.model_validate(base), db)
        assert isinstance(default.detector, SceneDetectShots)
        cfg = AppConfig.model_validate(
            {**base, "shots": {"detector": "transnetv2", "threshold": 0.4}}
        )
        chosen = build_providers(cfg, db)
        assert isinstance(chosen.detector, TransNetV2Shots) and chosen.detector.threshold == 0.4
    finally:
        db.close()
    with pytest.raises(ValueError):
        AppConfig.model_validate({**base, "shots": {"detector": "magic"}})
    with pytest.raises(ValueError):
        AppConfig.model_validate({**base, "shots": {"threshold": 1.5}})
