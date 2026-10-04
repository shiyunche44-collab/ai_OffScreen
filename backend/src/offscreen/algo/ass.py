"""ASS subtitle text from timeline subtitle items. Pure: the renderer writes the file."""

from __future__ import annotations

from collections.abc import Sequence

from offscreen.domain.common import Rational
from offscreen.domain.timeline import SubtitleItem

FONT = "Noto Sans CJK SC"
"""Looked up through fontconfig; libass falls back to any installed CJK font."""
FONT_HEIGHT_RATIO = 0.05
"""Font size as a share of the picture height (54 px at 1080p)."""
BOTTOM_MARGIN_RATIO = 0.07
"""Keeps lines inside the platforms' safe area."""
SIDE_MARGIN_RATIO = 0.05


def ass_time(frame: int, fps: Rational) -> str:
    """`H:MM:SS.cc` of the start of `frame` (ASS has centisecond resolution)."""
    cs = (frame * 100 * fps.den * 2 + fps.num) // (2 * fps.num)
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _escape(text: str) -> str:
    return (
        text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", "\\N")
    )  # ASS override syntax cannot be escaped, so look-alike characters stand in


def build_ass(
    items: Sequence[SubtitleItem], fps: Rational, width: int, height: int, font: str = FONT
) -> str:
    size = max(12, round(height * FONT_HEIGHT_RATIO))
    margin_v = round(height * BOTTOM_MARGIN_RATIO)
    margin_h = round(width * SIDE_MARGIN_RATIO)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"0,0,0,0,100,100,0,0,1,{max(1.0, size / 24):.1f},0,2,{margin_h},{margin_h},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for it in items:
        start, end = ass_time(it.f0, fps), ass_time(it.f1, fps)
        if start == end:  # a one-frame line shorter than a centisecond cannot happen, but be safe
            continue
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{_escape(it.text)}")
    return "\n".join(lines) + "\n"
