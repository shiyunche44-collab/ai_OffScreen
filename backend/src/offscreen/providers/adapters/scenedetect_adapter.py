"""ShotDetector backed by PySceneDetect's AdaptiveDetector (M1; TransNetV2 comes in M3)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from scenedetect import AdaptiveDetector, SceneManager, open_video

from offscreen.domain.common import TimeRange
from offscreen.providers.ports import DetectionCanceled

# Raise `VERSION` when the settings below change: it is part of the cache key.
VERSION = 1
ADAPTIVE_THRESHOLD = 3.0
MIN_CONTENT_VAL = 15.0


class SceneDetectShots:
    """Cuts only; fades and dissolves are mostly missed, which post-processing tolerates."""

    id = f"scenedetect/adaptive@{VERSION}:t{ADAPTIVE_THRESHOLD}:c{MIN_CONTENT_VAL}"

    def detect(
        self,
        video: Path,
        *,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[TimeRange]:
        stream = open_video(str(video))
        manager = SceneManager()
        # min_scene_len=1: short fragments are merged later, with the same rule for
        # every detector, rather than being silently swallowed here.
        manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=ADAPTIVE_THRESHOLD,
                min_content_val=MIN_CONTENT_VAL,
                min_scene_len=1,
            )
        )
        total = stream.duration.frame_num if stream.duration is not None else 0
        canceled = False

        def on_frame(_img: Any, tc: Any) -> None:
            nonlocal canceled
            if should_cancel is not None and should_cancel():
                canceled = True
                manager.stop()
            elif on_progress is not None and total:
                on_progress(min(1.0, tc.frame_num / total))

        manager.detect_scenes(stream, callback=on_frame)
        if canceled:
            raise DetectionCanceled("shot detection canceled")
        fps = stream.frame_rate
        return [
            TimeRange(
                start_ms=round(a.frame_num * 1000 / fps),
                end_ms=round(b.frame_num * 1000 / fps),
            )
            for a, b in manager.get_scene_list()
            if b.frame_num > a.frame_num
        ]
