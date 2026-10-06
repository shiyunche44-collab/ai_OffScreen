"""Built-in writing style presets (`builtin/*.yaml`), loaded into `StylePreset`."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import ValidationError

from offscreen.domain.style import StylePreset

_DIR = Path(__file__).parent / "builtin"


class StyleError(ValueError):
    pass


@cache
def _load_all() -> dict[str, StylePreset]:
    presets: dict[str, StylePreset] = {}
    for path in sorted(_DIR.glob("*.yaml")):
        try:
            preset = StylePreset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (OSError, yaml.YAMLError, ValidationError) as e:
            raise StyleError(f"style preset {path.name} is invalid: {e}") from e
        if preset.id != path.stem:
            raise StyleError(f"style preset {path.name}: id {preset.id!r} must match the file name")
        presets[preset.id] = preset
    return presets


def ids() -> list[str]:
    return sorted(_load_all())


def get(style_id: str) -> StylePreset:
    try:
        return _load_all()[style_id]
    except KeyError:
        raise StyleError(f"unknown style {style_id!r}; known: {', '.join(ids())}") from None


def all_presets() -> list[StylePreset]:
    return [_load_all()[i] for i in ids()]
