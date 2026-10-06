"""Looking at the writing style presets."""

from __future__ import annotations

from offscreen import styles
from offscreen.domain.style import StylePreset
from offscreen.services.errors import NotFound


def list_styles() -> list[StylePreset]:
    return styles.all_presets()


def get_style(style_id: str) -> StylePreset:
    try:
        return styles.get(style_id)
    except styles.StyleError as e:
        raise NotFound(str(e)) from e


def format_style(preset: StylePreset) -> str:
    lines = [
        f"{preset.id}  {preset.name}",
        preset.description,
        "",
        f"语气：{preset.tone}",
        f"人称：{'第一人称' if preset.perspective == 'first' else '第三人称'}",
        "结构：",
        *(f"  {b.share:>4.0%}  {b.name:<12} {b.purpose}" for b in preset.structure),
        "开头钩子：",
        *(f"  - {h}" for h in preset.hook_types),
    ]
    if preset.phrases:
        lines += ["常用句式：", *(f"  - {p}" for p in preset.phrases)]
    if preset.banned_words:
        lines.append(f"禁用词：{'、'.join(preset.banned_words)}")
    return "\n".join(lines)
