"""Throttled progress reporting: stages may call `progress()` thousands of times per second
(a loop over frames), but the database sees at most one write per interval."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

DEFAULT_INTERVAL_S = 0.5
"""At most two writes per second (IMPLEMENTATION_PLAN M2-02)."""


class ProgressThrottle:
    """Keeps the latest report and writes it when the interval has passed. A report that arrives
    too early is not lost: it waits as pending until the next report or `flush()` (the worker
    flushes on every tick and before recording a job's outcome)."""

    def __init__(
        self,
        write: Callable[[float, str], None],
        min_interval_s: float = DEFAULT_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._write = write
        self._interval = min_interval_s
        self._clock = clock
        self._lock = threading.Lock()
        self._last_written: float | None = None
        self._last_value: tuple[float, str] | None = None
        self._pending: tuple[float, str] | None = None

    def report(self, frac: float, msg: str = "") -> None:
        with self._lock:
            self._pending = (frac, msg)
            self._flush_if_due()

    def flush(self, *, force: bool = False) -> None:
        """Write the pending report if the interval passed (always, with `force`)."""
        with self._lock:
            self._flush_if_due(force=force)

    def _flush_if_due(self, *, force: bool = False) -> None:
        if self._pending is None:
            return
        now = self._clock()
        if (
            not force
            and self._last_written is not None
            and now - self._last_written < self._interval
        ):
            return
        value, self._pending = self._pending, None
        if value == self._last_value:
            return  # nothing new to tell the database
        self._write(*value)
        self._last_written, self._last_value = now, value
