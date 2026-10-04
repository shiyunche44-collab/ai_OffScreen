"""ffprobe -> domain values."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pydantic import BaseModel

from offscreen.domain.asset import AudioStream, VideoInfo
from offscreen.domain.common import Rational


class ProbeError(RuntimeError):
    pass


class ProbeResult(BaseModel):
    duration_ms: int
    video: VideoInfo
    audio: list[AudioStream]


def _fps(rate: str) -> Rational:
    num, _, den = rate.partition("/")
    n, d = int(num), int(den or 1)
    if n <= 0 or d <= 0:
        raise ProbeError(f"unusable frame rate {rate!r}")
    return Rational(num=n, den=d)


def probe(path: Path, binary: str = "ffprobe") -> ProbeResult:
    proc = subprocess.run(
        [
            binary,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise ProbeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    videos = [
        s
        for s in streams
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
    ]
    if not videos:
        raise ProbeError(f"no video stream in {path}")
    v = videos[0]
    duration = info.get("format", {}).get("duration") or v.get("duration")
    if duration is None:
        raise ProbeError(f"cannot determine duration of {path}")
    audio = [
        AudioStream(
            index=int(s["index"]),
            channels=int(s.get("channels", 1)),
            sample_rate=int(s.get("sample_rate", 48000)),
            language=s.get("tags", {}).get("language"),
        )
        for s in streams
        if s.get("codec_type") == "audio"
    ]
    return ProbeResult(
        duration_ms=round(float(duration) * 1000),
        video=VideoInfo(
            width=int(v["width"]),
            height=int(v["height"]),
            fps=_fps(
                v.get("avg_frame_rate")
                if v.get("avg_frame_rate", "0/0") not in ("0/0", "0/1")
                else v["r_frame_rate"]
            ),
            codec=v["codec_name"],
        ),
        audio=audio,
    )
