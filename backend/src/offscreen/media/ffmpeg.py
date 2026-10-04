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
