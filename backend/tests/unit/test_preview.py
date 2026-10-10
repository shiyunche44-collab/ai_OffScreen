"""Segment preview: the output size and what the cache key does and does not depend on."""

from __future__ import annotations

import pytest

from offscreen.algo.plan_build import text_digest
from offscreen.domain.asset import AudioStream, MediaAsset, VideoInfo
from offscreen.domain.common import Rational
from offscreen.domain.plan import AudioRef, Clip, EditPlan, PlanSegment, ScriptRef, VoiceSpec
from offscreen.stages.output.preview import (
    PreviewError,
    preview_spec,
    render_segment_preview,
    segment_hash,
)


def asset(
    width: int = 1920, height: int = 1080, fingerprint: str = "sha256:" + "a" * 64
) -> MediaAsset:
    return MediaAsset(
        id="ast_t1",
        title="t",
        source_path="/x.mkv",
        fingerprint=fingerprint,
        duration_ms=60_000,
        video=VideoInfo(
            width=width, height=height, fps=Rational(num=24000, den=1001), codec="h264"
        ),
        audio=[AudioStream(index=1, channels=2, sample_rate=48000)],
    )


def seg(i: int, text: str = "你好", shot_in: int = 0) -> PlanSegment:
    return PlanSegment(
        id=f"seg_{i:02d}",
        kind="narration",
        text=text,
        text_hash=text_digest(text),
        voice=VoiceSpec(voice_id="v"),
        audio=AudioRef(file=f"tts/{i}.wav", duration_ms=1000),
        clips=[Clip(asset_id="ast_t1", src_in_ms=shot_in, src_out_ms=shot_in + 1000)],
    )


def plan(*segments: PlanSegment, version: int = 1) -> EditPlan:
    return EditPlan(
        id="pln_t",
        project_id="prj_t",
        version=version,
        author="ai",
        script_ref=ScriptRef(id="scr_t", version=1),
        segments=list(segments),
    )


def test_the_preview_is_360p_with_the_films_shape() -> None:
    spec = preview_spec(asset())

    assert (spec.width, spec.height, spec.profile) == (640, 360, "preview")
    assert spec.fps == Rational(num=24000, den=1001) and spec.layout == "keep"


def test_odd_sizes_come_out_even_and_a_small_film_is_not_enlarged() -> None:
    assert (preview_spec(asset(1000, 563)).width % 2, preview_spec(asset(1000, 563)).height) == (
        0,
        360,
    )
    small = preview_spec(asset(320, 180))
    assert (small.width, small.height) == (320, 180)


def test_the_hash_follows_the_segment_not_the_plan_around_it() -> None:
    a = asset()
    base = segment_hash(plan(seg(1), seg(2)), "seg_01", a, "p")

    # other segments, the version and the order do not matter
    assert segment_hash(plan(seg(2), seg(1), seg(3), version=9), "seg_01", a, "p") == base
    # the segment itself, the film and the proxy do
    assert segment_hash(plan(seg(1, text="改了")), "seg_01", a, "p") != base
    assert segment_hash(plan(seg(1, shot_in=500)), "seg_01", a, "p") != base
    assert (
        segment_hash(plan(seg(1)), "seg_01", asset(fingerprint="sha256:" + "b" * 64), "p") != base
    )
    assert segment_hash(plan(seg(1)), "seg_01", a, "other-proxy") != base
    assert segment_hash(plan(seg(1), seg(2)), "seg_02", a, "p") != base


def test_an_unknown_or_stale_segment_is_refused_before_anything_is_rendered(
    tmp_path: object,
) -> None:
    from pathlib import Path

    d = Path(str(tmp_path))
    kw = {
        "plan_dir": d,
        "asset": asset(),
        "proxy": d / "none.mp4",
        "proxy_hash": "p",
        "audio": None,
        "previews_dir": d / "previews",
    }
    with pytest.raises(PreviewError, match="no segment seg_09"):
        render_segment_preview(plan(seg(1)), "seg_09", **kw)  # type: ignore[arg-type]
    stale = seg(1).model_copy(update={"stale": True})
    with pytest.raises(PreviewError, match="stale"):
        render_segment_preview(plan(stale), "seg_01", **kw)  # type: ignore[arg-type]
    with pytest.raises(PreviewError, match="proxy film not found"):
        render_segment_preview(plan(seg(1)), "seg_01", **kw)  # type: ignore[arg-type]
