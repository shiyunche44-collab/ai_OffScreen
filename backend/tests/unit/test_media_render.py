from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from offscreen.domain.common import Rational
from offscreen.media.render import (
    AudioPart,
    _concat_quote,
    clip_filter,
    concat_videos,
    encode_final,
    escape_filter_value,
    even,
    mix_audio,
    mix_graph,
    render_clip,
)

FPS = Rational(num=24000, den=1001)


def probe(path: Path) -> dict:  # type: ignore[type-arg]
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
         "-print_format", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    return json.loads(out)  # type: ignore[no-any-return]


def video_stream(info: dict) -> dict:  # type: ignore[type-arg]
    return next(s for s in info["streams"] if s["codec_type"] == "video")


# ---- string builders (no ffmpeg needed) ---------------------------------------------------
def test_even() -> None:
    assert [even(n) for n in (1, 2, 3, 1080, 853)] == [2, 2, 2, 1080, 852]


def test_clip_filter() -> None:
    f = clip_filter(FPS, 853, 480, 1.0, 36)
    assert f == (
        "fps=24000/1001,scale=852:480:flags=lanczos,setsar=1,format=yuv420p,"
        "tpad=stop_mode=clone:stop=36"
    )
    assert clip_filter(FPS, 640, 360, 1.25, 10).startswith("setpts=PTS/1.250000,fps=")


def test_mix_graph_places_gains_and_fades() -> None:
    parts = [
        AudioPart(Path("n.wav"), start_ms=1500, gain_db=0.0),
        AudioPart(
            Path("f.mkv"), start_ms=0, gain_db=-20.0, seek_ms=5000, duration_ms=1000, fade=True
        ),
    ]
    g = mix_graph(parts, 4000)
    assert (
        "[0:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=0.00dB,adelay=1500:all=1[a0]"
        in g
    )
    assert "volume=-20.00dB,afade=t=in:d=0.01,afade=t=out:st=0.990:d=0.01,adelay=0:all=1[a1]" in g
    assert g.endswith(
        "[a0][a1]amix=inputs=2:normalize=0:duration=longest,apad=whole_dur=4.000,atrim=end=4.000[out]"
    )
    with pytest.raises(ValueError):
        mix_graph([], 1000)


def test_filter_and_concat_quoting() -> None:
    assert escape_filter_value("/tmp/a b/it's:x,y[1].ass") == "/tmp/a b/it\\'s\\:x\\,y\\[1\\].ass"
    assert _concat_quote(Path("/tmp/it's.mp4")) == "'/tmp/it'\\''s.mp4'"


# ---- real ffmpeg --------------------------------------------------------------------------
def test_clip_has_exactly_the_requested_frames_even_past_the_end_of_the_source(
    clip: Path, tmp_path: Path
) -> None:
    ok = tmp_path / "ok.mp4"
    render_clip(clip, ok, src_in_ms=500, frames=24, fps=FPS, width=320, height=180)
    v = video_stream(probe(ok))
    assert (v["nb_read_frames"], v["width"], v["height"], v["pix_fmt"]) == (
        "24",
        320,
        180,
        "yuv420p",
    )
    assert not [s for s in probe(ok)["streams"] if s["codec_type"] == "audio"]

    # Starting 0.5 s before the source ends but asking for 1 s: the last frame is held.
    short = tmp_path / "short.mp4"
    render_clip(clip, short, src_in_ms=2500, frames=24, fps=FPS, width=320, height=180)
    assert video_stream(probe(short))["nb_read_frames"] == "24"


def test_clip_resizes_and_retimes(clip: Path, tmp_path: Path) -> None:
    dst = tmp_path / "r.mp4"
    render_clip(
        clip,
        dst,
        src_in_ms=0,
        frames=10,
        fps=Rational(num=25, den=1),
        width=161,
        height=91,
        speed=2.0,
    )
    v = video_stream(probe(dst))
    assert (v["nb_read_frames"], v["width"], v["height"], v["r_frame_rate"]) == (
        "10",
        160,
        90,
        "25/1",
    )


def test_concat_mix_and_final_encode(clip: Path, tmp_path: Path) -> None:
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    render_clip(clip, a, src_in_ms=0, frames=12, fps=FPS, width=320, height=180)
    render_clip(clip, b, src_in_ms=1000, frames=12, fps=FPS, width=320, height=180)
    silent = tmp_path / "silent.mp4"
    concat_videos([a, b], silent, list_file=tmp_path / "l.txt")
    assert video_stream(probe(silent))["nb_read_frames"] == "24"

    total_ms = round(24 * 1001 / 24)  # 24 frames at 23.976 fps
    mix = tmp_path / "mix.wav"
    mix_audio(
        [
            AudioPart(clip, start_ms=0, gain_db=-20.0, seek_ms=500, duration_ms=400, fade=True),
            AudioPart(clip, start_ms=500, gain_db=0.0, seek_ms=0, duration_ms=300),
        ],
        mix, total_ms=total_ms, script_file=tmp_path / "mix.filter",
    )  # fmt: skip
    a_info = probe(mix)["streams"][0]
    assert (a_info["sample_rate"], a_info["channels"]) == ("48000", 2)
    assert abs(float(probe(mix)["format"]["duration"]) * 1000 - total_ms) < 2

    final = tmp_path / "final.mp4"
    encode_final(silent, mix, final, subtitles=None, duration_ms=total_ms)
    info = probe(final)
    kinds = {s["codec_type"]: s for s in info["streams"]}
    assert kinds["video"]["nb_read_frames"] == "24" and kinds["audio"]["codec_name"] == "aac"
    assert abs(float(info["format"]["duration"]) * 1000 - total_ms) < 80
