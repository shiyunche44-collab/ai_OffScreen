"""The only place that spawns ffmpeg (ARCHITECTURE rule R5)."""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence

ProgressFn = Callable[[float], None]
STDERR_TAIL_LINES = 60


class FFmpegError(RuntimeError):
    def __init__(self, message: str, *, returncode: int | None, cmd: Sequence[str], stderr: str):
        super().__init__(f"{message}\ncmd: {' '.join(cmd)}\n--- stderr (tail) ---\n{stderr}")
        self.returncode = returncode
        self.cmd = list(cmd)
        self.stderr = stderr


class FFmpegCanceled(RuntimeError):
    pass


def run_ffmpeg(
    args: Sequence[str],
    *,
    duration_ms: int | None = None,
    on_progress: ProgressFn | None = None,
    timeout_s: float | None = None,
    should_cancel: Callable[[], bool] | None = None,
    binary: str = "ffmpeg",
) -> None:
    """Run ffmpeg with `-progress` parsing. `args` excludes the binary and global flags.

    Progress is reported as a 0..1 fraction when `duration_ms` is known. Raises
    FFmpegError (non-zero exit, timeout) with the stderr tail, or FFmpegCanceled."""
    cmd = [binary, "-hide_banner", "-nostdin", "-nostats", "-progress", "pipe:1", "-y", *args]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None and proc.stderr is not None
    tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
    drain = threading.Thread(target=lambda: tail.extend(proc.stderr or ()), daemon=True)
    drain.start()
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    killed: str | None = None

    def watchdog() -> None:
        nonlocal killed
        while proc.poll() is None:
            if deadline is not None and time.monotonic() > deadline:
                killed = "timeout"
            elif should_cancel is not None and should_cancel():
                killed = "cancel"
            if killed:
                proc.kill()
                return
            time.sleep(0.05)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            # `out_time_ms` is (despite the name) in microseconds in ffmpeg's -progress output.
            if key == "out_time_us" and on_progress and duration_ms and value.lstrip("-").isdigit():
                on_progress(min(max(int(value) / 1000 / duration_ms, 0.0), 1.0))
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        drain.join(timeout=2)
    stderr = "".join(tail)
    if killed == "cancel":
        raise FFmpegCanceled("ffmpeg canceled")
    if killed == "timeout":
        raise FFmpegError(
            f"ffmpeg timed out after {timeout_s}s",
            returncode=proc.returncode,
            cmd=cmd,
            stderr=stderr,
        )
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed (exit {proc.returncode})",
            returncode=proc.returncode,
            cmd=cmd,
            stderr=stderr,
        )
    if on_progress:
        on_progress(1.0)


def read_raw_video(
    video: str,
    *,
    width: int,
    height: int,
    duration_ms: int | None = None,
    on_progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
    binary: str = "ffmpeg",
    chunk_frames: int = 2048,
) -> bytes:
    """Every frame of the video as raw rgb24, scaled to `width` x `height`, one after another
    (frames are neither dropped nor repeated, so frame n is frame n of the film). The caller knows
    the size: `len(result) = frames * width * height * 3`. Progress is a 0..1 fraction from the
    frame timestamps when `duration_ms` is given. Raises FFmpegError or FFmpegCanceled."""
    cmd = [
        binary, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "error",
        "-i", video, "-map", "0:v:0", "-an", "-fps_mode", "passthrough",
        "-vf", f"scale={width}:{height}:flags=bilinear,format=rgb24",
        "-f", "rawvideo", "pipe:1",
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
    drain = threading.Thread(
        target=lambda: tail.extend(line.decode("utf-8", "replace") for line in proc.stderr or ()),
        daemon=True,
    )
    drain.start()
    frame_bytes = width * height * 3
    chunks: list[bytes] = []
    got = 0
    try:
        while True:
            if should_cancel is not None and should_cancel():
                proc.kill()
                raise FFmpegCanceled("ffmpeg canceled")
            chunk = proc.stdout.read(chunk_frames * frame_bytes)
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
            if on_progress and duration_ms:
                on_progress(min(1.0, got / frame_bytes / max(1.0, duration_ms / 1000 * 24)))
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        drain.join(timeout=2)
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed (exit {proc.returncode})",
            returncode=proc.returncode,
            cmd=cmd,
            stderr="".join(tail),
        )
    return b"".join(chunks)
