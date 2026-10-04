from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from offscreen.domain.asset import AudioStream, VideoInfo
from offscreen.domain.common import Rational
from offscreen.media.probe import ProbeError, ProbeResult
from offscreen.stages.analysis.ingest import IngestError, find_external_subtitles, ingest
from offscreen.store.db import Database
from offscreen.store.files import fingerprint_file
from offscreen.store.repos import AssetRepo


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[AssetRepo]:
    db = Database(tmp_path / "db" / "offscreen.db")
    yield AssetRepo(db)
    db.close()


def fake_probe(_: Path) -> ProbeResult:
    return ProbeResult(
        duration_ms=90_000,
        video=VideoInfo(width=1920, height=800, fps=Rational(num=24, den=1), codec="h264"),
        audio=[AudioStream(index=1, channels=2, sample_rate=48000, language="eng")],
    )


def movie(tmp_path: Path, name: str = "Sintel.mkv", data: bytes = b"movie-bytes") -> Path:
    p = tmp_path / "movies" / name
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(data)
    return p


# --- fingerprint -----------------------------------------------------------------------


def test_fingerprint_small_file_hashes_everything(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"x" * 100)
    b.write_bytes(b"x" * 99 + b"y")
    assert fingerprint_file(a, span=16) != fingerprint_file(b, span=16)
    assert fingerprint_file(a, span=16).startswith("sha256:")


def test_fingerprint_large_file_uses_size_head_and_tail_only(tmp_path: Path) -> None:
    span = 16
    base = b"H" * span + b"." * 100 + b"T" * span
    p = tmp_path / "big"
    p.write_bytes(base)
    fp = fingerprint_file(p, span=span)

    mid = tmp_path / "mid"  # same size/head/tail, different middle: not hashed
    mid.write_bytes(b"H" * span + b"!" * 100 + b"T" * span)
    assert fingerprint_file(mid, span=span) == fp

    for name, data in {
        "head": b"h" + base[1:],
        "tail": base[:-1] + b"t",
        "longer": base + b"extra-middle-doesnt-matter-but-size-does",
    }.items():
        q = tmp_path / name
        q.write_bytes(data)
        assert fingerprint_file(q, span=span) != fp, name


def test_fingerprint_is_independent_of_path(tmp_path: Path) -> None:
    a = movie(tmp_path, "a.mkv")
    b = movie(tmp_path, "renamed.mkv")
    assert fingerprint_file(a) == fingerprint_file(b)


# --- ingest ----------------------------------------------------------------------------


def test_ingest_registers_asset_with_probe_data(tmp_path: Path, repo: AssetRepo) -> None:
    p = movie(tmp_path)
    r = ingest(p, repo, probe_fn=fake_probe)
    assert r.created and not r.relinked
    a = r.asset
    assert a.id.startswith("ast_")
    assert a.title == "Sintel"
    assert a.source_path == str(p.resolve())
    assert a.fingerprint == fingerprint_file(p)
    assert (a.duration_ms, a.video.width, a.audio[0].language) == (90_000, 1920, "eng")
    assert a.derived.proxy is None and a.subtitles_external is None
    assert repo.get(a.id) == a


def test_same_file_twice_returns_the_same_asset(tmp_path: Path, repo: AssetRepo) -> None:
    p = movie(tmp_path)
    calls = []
    probe_counting = lambda f: (calls.append(f), fake_probe(f))[1]  # noqa: E731
    first = ingest(p, repo, probe_fn=probe_counting)
    again = ingest(p, repo, probe_fn=probe_counting)
    assert again.asset == first.asset and not again.created
    assert len(calls) == 1  # no re-probe
    assert len(repo.list()) == 1


def test_copy_of_the_file_maps_to_the_original_asset(tmp_path: Path, repo: AssetRepo) -> None:
    p = movie(tmp_path)
    first = ingest(p, repo, probe_fn=fake_probe)
    copy = movie(tmp_path, "Sintel copy.mkv")
    again = ingest(copy, repo, probe_fn=fake_probe)
    assert again.asset == first.asset and not again.relinked
    assert again.asset.source_path == str(p.resolve())  # original still exists: keep it


def test_moved_file_is_relinked_by_fingerprint(tmp_path: Path, repo: AssetRepo) -> None:
    p = movie(tmp_path)
    first = ingest(p, repo, probe_fn=fake_probe)
    new = tmp_path / "elsewhere" / "Sintel.mkv"
    new.parent.mkdir()
    shutil.move(p, new)
    new.with_suffix(".srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")

    r = ingest(new, repo, probe_fn=fake_probe)
    assert (r.created, r.relinked) == (False, True)
    assert r.asset.id == first.asset.id
    assert r.asset.source_path == str(new.resolve())
    assert r.asset.subtitles_external == str(new.with_suffix(".srt"))
    assert repo.get(first.asset.id) == r.asset


def test_external_subtitles_prefer_srt_then_ass(tmp_path: Path) -> None:
    p = movie(tmp_path)
    assert find_external_subtitles(p) is None
    p.with_suffix(".ass").write_text("x")
    assert find_external_subtitles(p) == str(p.with_suffix(".ass"))
    p.with_suffix(".srt").write_text("x")
    assert find_external_subtitles(p) == str(p.with_suffix(".srt"))


def test_errors_are_ingest_errors(tmp_path: Path, repo: AssetRepo) -> None:
    with pytest.raises(IngestError, match="not a file"):
        ingest(tmp_path / "missing.mkv", repo, probe_fn=fake_probe)
    with pytest.raises(IngestError, match="not a file"):
        ingest(tmp_path, repo, probe_fn=fake_probe)

    def bad_probe(_: Path) -> ProbeResult:
        raise ProbeError("no video stream")

    with pytest.raises(IngestError, match="no video"):
        ingest(movie(tmp_path), repo, probe_fn=bad_probe)
    assert repo.list() == []  # nothing registered on failure


def test_ingest_real_clip_with_ffprobe(clip: Path, tmp_path: Path, repo: AssetRepo) -> None:
    r = ingest(clip, repo)
    assert (r.asset.video.fps.num, r.asset.video.fps.den) == (24000, 1001)
    assert r.asset.title == "clip"
    assert ingest(clip, repo).asset.id == r.asset.id
