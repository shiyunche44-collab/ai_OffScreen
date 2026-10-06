"""Style presets: the built-in YAML files load, the schema rejects inconsistent ones, and the
CLI shows them."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from offscreen import styles
from offscreen.cli import app
from offscreen.domain.style import StylePreset


def preset_data(**over: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "demo",
        "name": "演示",
        "description": "用于测试",
        "tone": "平实",
        "structure": [
            {"name": "hook", "purpose": "开场", "share": 0.4},
            {"name": "ending", "purpose": "收尾", "share": 0.6},
        ],
        "hook_types": ["从一个问题开始"],
        "phrases": ["后来，……"],
        "banned_words": ["不得不说"],
    }
    data.update(over)
    return data


def test_the_three_built_in_styles_exist_and_are_valid() -> None:
    assert styles.ids() == ["emotional", "roast", "suspense"]
    assert [p.name for p in styles.all_presets()] == ["情感走心", "轻松吐槽", "悬疑紧凑"]
    for preset in styles.all_presets():
        assert preset.structure[0].name == "hook"  # every style opens with a hook
        assert preset.structure[-1].name == "ending"
        assert preset.hook_types and preset.phrases and preset.banned_words


def test_unknown_style_names_the_known_ones() -> None:
    with pytest.raises(styles.StyleError, match="suspense"):
        styles.get("nope")


def test_banned_in_reports_the_words_present_in_listed_order() -> None:
    preset = styles.get("suspense")
    text = "不得不说，这部电影讲述了一个故事"
    assert preset.banned_in(text) == ["这部电影讲述了", "不得不说"]
    assert preset.banned_in("他推开了门") == []


def test_valid_data_round_trips() -> None:
    preset = StylePreset.model_validate(preset_data())
    assert StylePreset.model_validate(preset.model_dump()) == preset


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"id": "Bad Id"}, "id"),
        ({"structure": []}, "structure"),
        (
            {"structure": [{"name": "a", "purpose": "x", "share": 0.5}]},
            "add up to 1",
        ),
        (
            {
                "structure": [
                    {"name": "a", "purpose": "x", "share": 0.5},
                    {"name": "a", "purpose": "y", "share": 0.5},
                ]
            },
            "unique",
        ),
        ({"hook_types": []}, "hook_types"),
        ({"phrases": ["a", "a"]}, "repeat"),
        ({"banned_words": [" "]}, "empty"),
        ({"phrases": ["不得不说，……"]}, "banned words"),
        ({"perspective": "second"}, "perspective"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_inconsistent_presets_are_rejected(over: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        StylePreset.model_validate(preset_data(**over))


def test_cli_lists_and_shows_styles() -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["style", "list"])
    assert listed.exit_code == 0
    assert "suspense" in listed.output and "悬疑紧凑" in listed.output

    shown = runner.invoke(app, ["style", "show", "roast"])
    assert shown.exit_code == 0
    assert "轻松吐槽" in shown.output and "禁用词：" in shown.output

    missing = runner.invoke(app, ["style", "show", "nope"])
    assert missing.exit_code == 1
    assert "unknown style" in missing.output
