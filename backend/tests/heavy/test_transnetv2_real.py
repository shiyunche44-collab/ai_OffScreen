"""The real TransNetV2 network on a film made of synthetic shots with hard cuts and dissolves
(needs the `transnetv2-pytorch` package, which ships the weights). Not a substitute for a scored
real film: the cuts here are far easier than a movie's."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from offscreen.algo.cuts import shot_cut_frames
from offscreen.domain.common import Rational
from offscreen.providers.adapters.scenedetect_adapter import SceneDetectShots
from offscreen.providers.adapters.transnetv2_shots import TransNetV2Shots

pytestmark = pytest.mark.heavy

FPS = Rational(num=24, den=1)


def lavfi(out: Path, source: str) -> None:
    spec = source if "=" in source else f"{source}="
    spec += ("" if spec.endswith("=") else ":") + "size=640x360:rate=24"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", spec,
            "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
        ],
        check=True,
    )  # fmt: skip


@pytest.fixture(scope="module")
def film(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """4 s each of four sources: a 0.75 s fade, a 0.75 s dissolve, then a hard cut.
    Transitions span frames 78-96 and 156-174; the hard cut is at frame 252."""
    d = tmp_path_factory.mktemp("film")
    names = ["testsrc2", "mandelbrot", "life=mold=10:ratio=0.1", "cellauto=full=1"]
    for i, src in enumerate(names):
        lavfi(d / f"s{i}.mp4", src)
    out = d / "film.mp4"
    graph = (
        "[0:v][1:v]xfade=transition=fade:duration=0.75:offset=3.25[a];"
        "[a][2:v]xfade=transition=dissolve:duration=0.75:offset=6.5[b];"
        "[b][3:v]concat=n=2:v=1:a=0[c]"
    )
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            *[x for i in range(4) for x in ("-i", str(d / f"s{i}.mp4"))],
            "-filter_complex", graph, "-map", "[c]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
        ],
        check=True,
    )  # fmt: skip
    return out


def test_transnetv2_finds_the_gradual_transitions_that_colour_difference_misses(film: Path) -> None:
    pytest.importorskip("transnetv2_pytorch")
    net = shot_cut_frames([s.start_ms for s in TransNetV2Shots().detect(film)], FPS)
    assert any(78 <= c <= 97 for c in net)  # inside the fade
    assert any(156 <= c <= 175 for c in net)  # inside the dissolve
    assert any(abs(c - 252) <= 2 for c in net)  # the hard cut
    assert len(net) == 3

    classic = shot_cut_frames([s.start_ms for s in SceneDetectShots().detect(film)], FPS)
    assert any(abs(c - 252) <= 2 for c in classic)  # the hard cut is easy for both
    assert not any(78 <= c <= 97 for c in classic)  # the fade is invisible to colour differences
    assert len(classic) < len(net)  # (it caught the end of the dissolve at best)
