import subprocess
import threading
from pathlib import Path

import pytest

from offscreen.media.ffmpeg import FFmpegCanceled, FFmpegError, run_ffmpeg
from offscreen.media.probe import ProbeError, probe


def test_probe_reports_exact_rational_fps(clip: Path) -> None:
    r = probe(clip)
    assert (r.video.width, r.video.height, r.video.codec) == (320, 180, "h264")
    assert (r.video.fps.num, r.video.fps.den) == (24000, 1001)
    assert 2900 <= r.duration_ms <= 3200
    assert r.audio[0].sample_rate == 48000


def test_probe_errors(tmp_path: Path) -> None:
    with pytest.raises(ProbeError):
        probe(tmp_path / "missing.mp4")
    audio_only = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=1", str(audio_only)],
        check=True,
    )
    with pytest.raises(ProbeError, match="no video"):
        probe(audio_only)


def test_progress_reaches_one_and_is_monotonic(clip: Path, tmp_path: Path) -> None:
    seen: list[float] = []
    run_ffmpeg(
        ["-i", str(clip), "-vf", "scale=160:90", "-an", str(tmp_path / "o.mp4")],
        duration_ms=probe(clip).duration_ms,
        on_progress=seen.append,
    )
    assert seen and seen[-1] == 1.0
    assert seen == sorted(seen) and all(0.0 <= x <= 1.0 for x in seen)


def test_failure_carries_stderr_tail(tmp_path: Path) -> None:
    with pytest.raises(FFmpegError) as e:
        run_ffmpeg(["-i", str(tmp_path / "nope.mp4"), str(tmp_path / "o.mp4")])
    assert e.value.returncode != 0 and "nope.mp4" in e.value.stderr


def test_timeout_kills_process(tmp_path: Path) -> None:
    with pytest.raises(FFmpegError, match="timed out"):
        run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=1280x720:rate=30",
                "-t",
                "600",
                "-c:v",
                "libx264",
                str(tmp_path / "o.mp4"),
            ],
            timeout_s=1,
        )


def test_cancel(tmp_path: Path) -> None:
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    with pytest.raises(FFmpegCanceled):
        run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=1280x720:rate=30",
                "-t",
                "600",
                "-c:v",
                "libx264",
                str(tmp_path / "o.mp4"),
            ],
            should_cancel=flag.is_set,
        )
