from __future__ import annotations

import json
import subprocess
import wave
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import pytest

from offscreen.engine import ArtifactStore, Engine, StageCanceled
from offscreen.media.probe import probe
from offscreen.stages.analysis.ingest import ingest
from offscreen.stages.analysis.proxy import AUDIO_16K, AUDIO_48K, PROXY_FILE, ProxyError, ProxyStage
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo


def make_clip(path: Path, size: str, seconds: int, audio: bool = True) -> Path:
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i"]
    cmd += [f"testsrc2=size={size}:rate=24:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}:sample_rate=44100"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "48"]
    cmd += ["-c:a", "aac", "-shortest"] if audio else ["-an"]
    subprocess.run([*cmd, str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def hd_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_clip(tmp_path_factory.mktemp("hd") / "hd.mp4", "1280x720", 4)


@pytest.fixture
def env(tmp_path: Path) -> Iterator[tuple[AssetRepo, Engine]]:
    db = Database(tmp_path / "db" / "offscreen.db")
    repo = AssetRepo(db)
    yield repo, Engine(ArtifactStore(tmp_path / "artifacts"), [ProxyStage(repo)])
    db.close()


def streams(path: Path) -> dict:  # type: ignore[type-arg]
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(out.stdout)


def keyframe_times(path: Path) -> list[float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    rows = [line.split(",") for line in out.stdout.split()]
    return sorted(float(t) for t, flags in rows if "K" in flags)


def test_proxy_is_540p_h264_with_half_second_gop_and_faststart(
    hd_clip: Path, env: tuple[AssetRepo, Engine]
) -> None:
    repo, engine = env
    asset = ingest(hd_clip, repo).asset
    art = engine.ensure("analysis.proxy", {"asset_id": asset.id})

    proxy = art.path(PROXY_FILE)
    info = probe(proxy)
    assert (info.video.width, info.video.height, info.video.codec) == (960, 540, "h264")
    assert abs(info.duration_ms - asset.duration_ms) < 200
    assert info.audio and info.audio[0].channels == 2

    kf = keyframe_times(proxy)
    gaps = {round(b - a, 3) for a, b in pairwise(kf)}
    assert gaps == {0.5}  # 24 fps -> every 12 frames, no scene-cut keyframes

    head = proxy.read_bytes()[: 64 * 1024]
    assert 0 <= head.find(b"moov") < head.find(b"mdat") or head.find(b"mdat") < 0


def test_audio_tracks_have_expected_format(hd_clip: Path, env: tuple[AssetRepo, Engine]) -> None:
    repo, engine = env
    asset = ingest(hd_clip, repo).asset
    art = engine.ensure("analysis.proxy", {"asset_id": asset.id})
    for name, rate, ch in ((AUDIO_16K, 16000, 1), (AUDIO_48K, 48000, 2)):
        with wave.open(str(art.path(name))) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (rate, ch, 2)
            assert abs(w.getnframes() / rate - asset.duration_ms / 1000) < 0.25
    assert art.meta == {"has_audio": True}
    assert [f.path for f in art.manifest.files] == [AUDIO_16K, AUDIO_48K, PROXY_FILE]


def test_small_source_is_not_upscaled(clip: Path, env: tuple[AssetRepo, Engine]) -> None:
    repo, engine = env
    asset = ingest(clip, repo).asset  # 320x180
    art = engine.ensure("analysis.proxy", {"asset_id": asset.id})
    info = probe(art.path(PROXY_FILE))
    assert (info.video.width, info.video.height) == (320, 180)
    assert (info.video.fps.num, info.video.fps.den) == (24000, 1001)


def test_second_ensure_is_a_cache_hit_and_progress_is_monotonic(
    hd_clip: Path, tmp_path: Path
) -> None:
    db = Database(tmp_path / "db" / "o.db")
    repo = AssetRepo(db)
    events: list[float] = []
    store = ArtifactStore(tmp_path / "artifacts")
    engine = Engine(store, [ProxyStage(repo)], progress=lambda _s, f, _m: events.append(f))
    asset = ingest(hd_clip, repo).asset

    first = engine.ensure("analysis.proxy", {"asset_id": asset.id})
    assert events == sorted(events) and events[-1] == 1.0 and len(events) > 3
    n = len(events)
    again = engine.ensure("analysis.proxy", {"asset_id": asset.id})
    assert again.dir == first.dir and len(events) == n  # nothing re-encoded
    db.close()


def test_video_without_audio_yields_proxy_only(
    tmp_path: Path, env: tuple[AssetRepo, Engine]
) -> None:
    repo, engine = env
    silent = make_clip(tmp_path / "silent.mp4", "320x180", 2, audio=False)
    asset = ingest(silent, repo).asset
    art = engine.ensure("analysis.proxy", {"asset_id": asset.id})
    assert [f.path for f in art.manifest.files] == [PROXY_FILE]
    assert art.meta == {"has_audio": False}
    assert not probe(art.path(PROXY_FILE)).audio


def test_unknown_asset_and_missing_source(
    tmp_path: Path, clip: Path, env: tuple[AssetRepo, Engine]
) -> None:
    repo, engine = env
    with pytest.raises(ProxyError, match="unknown asset"):
        engine.ensure("analysis.proxy", {"asset_id": "ast_nope"})

    copy = tmp_path / "gone.mp4"
    copy.write_bytes(clip.read_bytes())
    asset = ingest(copy, repo).asset
    copy.unlink()
    with pytest.raises(ProxyError, match="missing"):
        engine.ensure("analysis.proxy", {"asset_id": asset.id})


def test_cancel_stops_encoding_and_leaves_no_artifact(hd_clip: Path, tmp_path: Path) -> None:
    db = Database(tmp_path / "db" / "o.db")
    repo = AssetRepo(db)
    store = ArtifactStore(tmp_path / "artifacts")
    asset = ingest(hd_clip, repo).asset
    ticks = {"n": 0}

    def cancel() -> bool:
        ticks["n"] += 1
        return ticks["n"] > 3  # not before the stage starts, but soon after

    engine = Engine(store, [ProxyStage(repo)], is_canceled=cancel)
    with pytest.raises(StageCanceled):
        engine.ensure("analysis.proxy", {"asset_id": asset.id})
    assert list((store.root / "analysis.proxy").iterdir()) == []
    db.close()
