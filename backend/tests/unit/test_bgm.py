"""The music library: registry, tags, scanning, and the renderer using a chosen track."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from offscreen.cli import app
from offscreen.config import AppConfig
from offscreen.domain.bgm import BgmLibrary, BgmTrack
from offscreen.media.audio import MIX_RATE, write_wav
from offscreen.services.bgm import BgmService
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.store.bgm import BgmStore


def music(path: Path, seconds: float = 2.0, amp: float = 0.3) -> Path:
    t = np.arange(round(seconds * MIX_RATE)) / MIX_RATE
    x = (np.sin(2 * np.pi * 330 * t) * amp).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, np.stack([x, x], axis=1))
    return path


@pytest.fixture
def svc(tmp_path: Path) -> BgmService:
    return BgmService(
        AppConfig.model_validate(
            {"data_dir": str(tmp_path / "data"), "tts": {"provider": "edge_tts"}}
        )
    )


def folder(svc: BgmService) -> Path:
    return svc.store.dir


def test_track_ids_and_files_are_unique() -> None:
    t = BgmTrack(id="a", name="A", file="a.wav", duration_ms=1000, sha256="0" * 64)
    with pytest.raises(ValidationError):
        BgmLibrary(tracks=[t, t.model_copy(update={"file": "b.wav"})])
    with pytest.raises(ValidationError):
        BgmLibrary(tracks=[t, t.model_copy(update={"id": "b"})])
    with pytest.raises(ValidationError):
        BgmTrack(id="has space", name="A", file="a.wav", duration_ms=1000, sha256="0" * 64)


def test_add_measures_the_file_and_normalizes_tags(svc: BgmService) -> None:
    music(folder(svc) / "tense_01.wav", seconds=2.0)
    t = svc.add("tense_01.wav", moods=["Tense", " dark ", "tense", ""], tempo="fast")
    assert (t.id, t.file, t.name, t.tempo) == ("tense_01", "tense_01.wav", "tense_01", "fast")
    assert t.moods == ["tense", "dark"] and abs(t.duration_ms - 2000) < 50
    assert len(t.sha256) == 64 and svc.get("tense_01") == t
    assert [x.id for x in svc.list(mood="DARK")] == ["tense_01"] and svc.list(mood="warm") == []
    assert svc.list(tempo="slow") == []


def test_adding_again_changes_tags_but_keeps_the_rest(svc: BgmService) -> None:
    music(folder(svc) / "a.wav")
    first = svc.add("a.wav", moods=["warm"], tempo="slow")
    again = svc.add("a.wav", name="Home")
    assert again.id == first.id and again.moods == ["warm"] and again.tempo == "slow"
    assert again.name == "Home" and len(svc.list()) == 1
    assert svc.add("a.wav", moods=[]).moods == []  # an empty list clears the tags


def test_a_changed_file_gets_a_new_checksum(svc: BgmService) -> None:
    f = music(folder(svc) / "a.wav", seconds=1.0)
    before = svc.add("a.wav")
    music(f, seconds=3.0)
    after = svc.add("a.wav")
    assert after.sha256 != before.sha256 and after.duration_ms > before.duration_ms


def test_bad_input_is_refused(svc: BgmService) -> None:
    folder(svc).mkdir(parents=True)
    (folder(svc) / "notes.txt").write_text("hi")
    for bad in ("missing.wav", "notes.txt", "../escape.wav", "/etc/passwd"):
        with pytest.raises(InvalidInput):
            svc.add(bad)
    music(folder(svc) / "a.wav")
    music(folder(svc) / "b.wav")
    svc.add("a.wav", track_id="same")
    with pytest.raises(InvalidInput, match="taken"):
        svc.add("b.wav", track_id="same")
    with pytest.raises(NotFound):
        svc.get("nope")
    with pytest.raises(NotFound):
        svc.remove("nope")


def test_scan_registers_new_music_only_and_ids_never_clash(svc: BgmService) -> None:
    music(folder(svc) / "a.wav")
    music(folder(svc) / "sub" / "a.wav")  # same stem, other folder
    (folder(svc) / "readme.txt").write_text("not music")
    (folder(svc) / "broken.mp3").write_bytes(b"not really mp3")
    added = svc.scan()
    assert sorted(t.id for t in added) == ["a", "a_2"]
    assert {t.file for t in added} == {"a.wav", "sub/a.wav"}
    assert svc.scan() == []  # nothing new the second time


def test_remove_keeps_the_file(svc: BgmService) -> None:
    f = music(folder(svc) / "a.wav")
    svc.add("a.wav")
    svc.remove("a")
    assert svc.list() == [] and f.is_file()


def test_the_library_digest_follows_the_music(svc: BgmService, tmp_path: Path) -> None:
    store = BgmStore(tmp_path / "data")
    empty = store.digest()
    music(folder(svc) / "a.wav")
    svc.add("a.wav", moods=["x"])
    one = store.digest()
    assert one != empty
    svc.add("a.wav", moods=["y"])  # tags are not music
    assert store.digest() == one
    music(folder(svc) / "a.wav", seconds=3.0)
    svc.add("a.wav")
    assert store.digest() != one


def test_the_command_line_lists_scans_tags_and_removes(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"data_dir: {tmp_path / 'data'}\ntts:\n  provider: edge_tts\n", encoding="utf-8"
    )
    music(tmp_path / "data" / "library" / "bgm" / "calm.wav")
    run = CliRunner()

    def cli(*args: str, ok: bool = True) -> str:
        r = run.invoke(app, ["bgm", *args, "--config", str(config)])
        assert (r.exit_code == 0) is ok, r.output
        return r.output

    assert cli("list").count("calm") == 0 and "no tracks" in cli("list")
    assert "1 new track" in cli("scan")
    out = cli("add", "calm.wav", "--mood", "warm", "--mood", "calm", "--tempo", "slow")
    assert "calm" in out and "slow" in out and "warm,calm" in out
    assert "calm" in cli("list", "--mood", "warm")
    assert "no tracks" in cli("list", "--mood", "tense")
    assert "error" in cli("add", "gone.wav", ok=False)
    assert "removed calm" in cli("remove", "calm")
    assert "no tracks" in cli("list")
