from __future__ import annotations

from offscreen.algo.ass import ass_time, build_ass
from offscreen.domain.common import Rational
from offscreen.domain.timeline import SubtitleItem

FPS = Rational(num=24000, den=1001)


def test_ass_time_rounds_to_centiseconds() -> None:
    assert ass_time(0, Rational(num=25, den=1)) == "0:00:00.00"
    assert ass_time(25, Rational(num=25, den=1)) == "0:00:01.00"
    assert ass_time(1, FPS) == "0:00:00.04"  # 41.7 ms
    assert ass_time(24 * 3600 * 25, Rational(num=25, den=1)) == "24:00:00.00"
    assert ass_time(90 * 25, Rational(num=25, den=1)) == "0:01:30.00"


def test_build_ass_structure_and_escaping() -> None:
    items = [
        SubtitleItem(f0=0, f1=25, text="你好，世界"),
        SubtitleItem(f0=25, f1=50, text="a{b}c\\d\ne"),
    ]
    ass = build_ass(items, Rational(num=25, den=1), 1920, 800, "Some Font")
    assert "PlayResX: 1920" in ass and "PlayResY: 800" in ass
    assert "Style: Default,Some Font,40," in ass  # 5% of 800
    lines = [x for x in ass.splitlines() if x.startswith("Dialogue:")]
    assert lines[0] == "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,你好，世界"
    # Override braces and backslashes cannot reach libass as syntax; newlines become \N.
    assert "{" not in lines[1].split(",,", 1)[1] and "\\d" not in lines[1]
    assert lines[1].endswith("a｛b｝c＼d\\Ne")


def test_build_ass_without_items_is_still_valid() -> None:
    ass = build_ass([], Rational(num=25, den=1), 640, 360)
    assert "[Events]" in ass and "Dialogue" not in ass
