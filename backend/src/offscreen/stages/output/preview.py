"""Preview of one plan segment (T4 + T5 on a single segment, deterministic: no model is called).

The segment is compiled on its own into a timeline and rendered small and fast from the proxy
film: 360p, x264 ultrafast, the same code as the final render (`render_timeline`) with other
encoder settings. Previews are cached by a hash of everything that shapes them (the segment as
stored, the film, the proxy, the output size, the encoder settings), so the same segment in any
plan version, or after an edit elsewhere in the plan, is not rendered again."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from offscreen.algo.ass import FONT
from offscreen.algo.compile import CompileError, compile_timeline
from offscreen.domain.asset import MediaAsset
from offscreen.domain.plan import EditPlan
from offscreen.domain.timeline import OutputSpec
from offscreen.media.ffmpeg import FFmpegCanceled
from offscreen.media.render import even
from offscreen.stages.output.compile import SUBTITLE_CHARS_LANDSCAPE, SUBTITLE_CHARS_PORTRAIT
from offscreen.stages.output.render import FINAL_FILE, Quality, RenderError, render_timeline

PREVIEW_HEIGHT = 360
PREVIEW_QUALITY = Quality(preset="ultrafast", clip_crf=30, final_crf=30)
PREVIEW_VERSION = 1
"""Bump when a preview of the same segment would look different."""


class PreviewError(RuntimeError):
    pass


@dataclass(frozen=True)
class SegmentPreview:
    file: Path
    cached: bool
    segment_hash: str
    duration_ms: int


def preview_spec(asset: MediaAsset) -> OutputSpec:
    """The film's picture shape at preview size (never larger than the film)."""
    v = asset.video
    height = min(PREVIEW_HEIGHT, v.height)
    width = even(round(v.width * height / v.height))
    return OutputSpec(profile="preview", width=width, height=even(height), fps=v.fps, layout="keep")


def segment_hash(
    plan: EditPlan, segment_id: str, asset: MediaAsset, proxy_hash: str, font: str = FONT
) -> str:
    seg = next((s for s in plan.segments if s.id == segment_id), None)
    if seg is None:
        raise PreviewError(f"the plan has no segment {segment_id}")
    payload = {
        "preview": PREVIEW_VERSION,
        "segment": seg.model_dump(mode="json"),
        "asset": asset.fingerprint,
        "proxy": proxy_hash,
        "output": preview_spec(asset).model_dump(mode="json"),
        "quality": [PREVIEW_QUALITY.preset, PREVIEW_QUALITY.clip_crf, PREVIEW_QUALITY.final_crf],
        "font": font,
        "subtitle_chars": [SUBTITLE_CHARS_LANDSCAPE, SUBTITLE_CHARS_PORTRAIT],
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def render_segment_preview(
    plan: EditPlan,
    segment_id: str,
    *,
    plan_dir: Path,
    asset: MediaAsset,
    proxy: Path,
    proxy_hash: str,
    audio: Path | None,
    previews_dir: Path,
    font: str = FONT,
    should_cancel: Callable[[], bool] | None = None,
) -> SegmentPreview:
    """The preview file of `segment_id` of `plan`, rendered unless one with the same hash is
    there. `plan_dir` is where the plan's narration files are, `proxy` / `audio` the proxy film
    and the film's sound (None: it has none)."""
    key = segment_hash(plan, segment_id, asset, proxy_hash, font)
    spec = preview_spec(asset)
    seg = next(s for s in plan.segments if s.id == segment_id)
    if seg.stale:
        raise PreviewError(f"{segment_id} is stale: build the plan first")
    single = plan.model_copy(update={"segments": [seg]})
    limit = SUBTITLE_CHARS_LANDSCAPE if spec.width >= spec.height else SUBTITLE_CHARS_PORTRAIT
    try:
        tl = compile_timeline(single, spec, subtitle_max_chars=limit)
    except CompileError as e:
        raise PreviewError(f"{segment_id}: {e}") from e
    duration_ms = round(tl.duration_frames * 1000 * spec.fps.den / spec.fps.num)

    final = previews_dir / f"{key}.mp4"
    if final.is_file():
        return SegmentPreview(final, True, key, duration_ms)
    if not proxy.is_file():
        raise PreviewError(f"proxy film not found: {proxy}")
    previews_dir.mkdir(parents=True, exist_ok=True)
    # Rendered beside the result and moved into place, so a reader never sees half a file and
    # two requests for the same preview do not clash.
    with tempfile.TemporaryDirectory(dir=previews_dir, prefix=".work-") as tmp:
        work = Path(tmp)
        out = work / "out"
        out.mkdir()
        try:
            render_timeline(
                tl,
                plan_dir,
                out,
                work,
                video=proxy,
                audio=audio,
                font=font,
                quality=PREVIEW_QUALITY,
                should_cancel=should_cancel,
            )
        except FFmpegCanceled as e:
            raise PreviewError("canceled") from e
        except RenderError as e:
            raise PreviewError(str(e)) from e
        os.replace(out / FINAL_FILE, final)
    return SegmentPreview(final, False, key, duration_ms)
