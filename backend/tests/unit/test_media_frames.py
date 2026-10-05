from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from offscreen.media.transcode import GRAY_WIDTH, extract_frame, make_sprite_sheets


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("media") / "c.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=24:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(p),
        ],
        check=True,
    )
    return p


def size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    w, h = out.strip().split(",")
    return int(w), int(h)


def test_extract_frame_can_also_write_the_gray_copy_in_one_decode(
    clip: Path, tmp_path: Path
) -> None:
    jpg, gray = tmp_path / "f.jpg", tmp_path / "f.gray"
    extract_frame(clip, jpg, at_ms=1500, height=180, gray_to=gray)
    assert size(jpg) == (320, 180)
    assert gray.stat().st_size == GRAY_WIDTH * 180  # 8-bit, 320 wide, height by aspect ratio


def test_extract_frame_without_gray_is_unchanged(clip: Path, tmp_path: Path) -> None:
    jpg = tmp_path / "f.jpg"
    extract_frame(clip, jpg, at_ms=0, height=360)
    assert size(jpg) == (640, 360)
    assert [p.name for p in tmp_path.iterdir()] == ["f.jpg"]


def test_sprite_sheets_are_filled_in_order_and_the_last_one_holds_the_rest(
    clip: Path, tmp_path: Path
) -> None:
    frames = []
    for i in range(5):
        f = tmp_path / f"f{i}.jpg"
        extract_frame(clip, f, at_ms=i * 500, height=180)
        frames.append(f)
    out = tmp_path / "out"
    out.mkdir()
    make_sprite_sheets(
        frames, out / "sheet_%03d.jpg", tile_width=160, tile_height=90, columns=2, rows=2
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "sheet_001.jpg",
        "sheet_002.jpg",
    ]  # no stray files
    assert size(out / "sheet_001.jpg") == (320, 180) == size(out / "sheet_002.jpg")


def test_tiles_letterbox_other_aspect_ratios(tmp_path: Path) -> None:
    wide = tmp_path / "wide.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=470x200",
            "-frames:v",
            "1",
            str(wide),
        ],
        check=True,
    )
    out = tmp_path / "s"
    out.mkdir()
    make_sprite_sheets(
        [wide], out / "s_%03d.jpg", tile_width=160, tile_height=90, columns=1, rows=1
    )
    px = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(out / "s_001.jpg"),
            "-vf",
            "crop=40:4:60:0,scale=1:1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    assert px[0] < 40 and px[1] < 40  # the top strip is black padding, not the picture


def test_packing_nothing_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        make_sprite_sheets(
            [], tmp_path / "s_%03d.jpg", tile_width=160, tile_height=90, columns=2, rows=2
        )
