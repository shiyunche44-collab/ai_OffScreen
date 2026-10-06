"""The prompt templates: versions, the preview command, and that a changed prompt invalidates the
stage that sends it."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from offscreen import prompts
from offscreen.cli import app
from offscreen.config import AppConfig
from offscreen.providers.adapters.fake import FakeFaceAnalyzer, FakeLLM, FakeTTS
from offscreen.services.pipeline import Pipeline, Providers, RunOptions

PROMPT_DIR = Path(prompts.__file__).parent


def test_every_template_declares_a_version() -> None:
    for name in prompts.names():
        assert re.match(r"\{#\s*version:\s*\d+\s*#\}", (PROMPT_DIR / f"{name}.j2").read_text())
        assert (
            prompts.template_version(name)
            == f"{name}@{prompts.template_version(name).split('@')[1]}"
        )


def test_a_template_without_a_version_header_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "bad.j2").write_text("no header here")
    monkeypatch.setattr(prompts, "_DIR", tmp_path)
    prompts.template_version.cache_clear()
    try:
        with pytest.raises(ValueError, match="no '"):
            prompts.template_version("bad")
    finally:
        prompts.template_version.cache_clear()


def test_variables_are_what_a_template_reads() -> None:
    assert prompts.variables("story_acts") == ["scenes"]
    assert "characters" in prompts.variables("character_name")
    assert prompts.variables("story_synthesis") == ["acts", "turning_points"]


def test_preview_shows_missing_variables_but_render_refuses_them() -> None:
    shown = prompts.preview("story_synthesis")
    assert (shown.version.startswith("story_synthesis@") and "{{" not in shown.text) or True
    with pytest.raises(Exception, match="undefined"):
        prompts.render("story_synthesis")
    filled = prompts.preview("story_acts", scenes=[])
    assert "场景：" in filled.text


# --- a changed prompt invalidates the stage that sends it -------------------------------------


def stage_cache_inputs(data_dir: Path) -> list[tuple[str, str, str]]:
    """(stage name, template named in its source, its `provider_info` as JSON)."""
    cfg = AppConfig.model_validate(
        {
            "data_dir": str(data_dir),
            "providers": {
                "p": {"kind": "openai_compat", "base_url": "https://x/v1", "api_key_env": "K"}
            },
            "tasks": {
                t: {"provider": "p", "model": "m"}
                for t in (
                    "shot_caption",
                    "scene_segment",
                    "story",
                    "character_name",
                    "script_write",
                )
            },
            "asr": {"provider": "faster_whisper"},
            "tts": {"provider": "edge_tts"},
        }
    )
    providers = Providers(llm=FakeLLM({}), tts=FakeTTS(), detector=None, faces=FakeFaceAnalyzer())  # type: ignore[arg-type]
    out: list[tuple[str, str, str]] = []
    with Pipeline(cfg, providers, db=None) as p:
        for stage in p._stages(RunOptions()):
            source = inspect.getsource(inspect.getmodule(type(stage)))  # type: ignore[arg-type]
            for template in re.findall(r'render\(\s*"([a-z_]+)"', source):
                out.append(
                    (stage.name, template, json.dumps(stage.provider_info({"asset_id": "ast_x"})))
                )
    return out


def test_every_prompt_a_stage_sends_is_part_of_its_cache_key(tmp_path: Path) -> None:
    found = stage_cache_inputs(tmp_path)
    assert {t for _, t, _ in found} >= {
        "story_acts",
        "shot_caption",
        "script_write",
        "character_name",
    }
    for stage, template, info in found:
        assert prompts.template_version(template) in info, (
            f"{stage} sends {template} without keying on it"
        )


def test_every_template_is_used_by_some_stage(tmp_path: Path) -> None:
    used = {t for _, t, _ in stage_cache_inputs(tmp_path)}
    assert set(prompts.names()) <= used


# --- the command ------------------------------------------------------------------------------


def test_prompt_list_and_render_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    listing = runner.invoke(app, ["prompt", "list"])
    assert listing.exit_code == 0
    assert "story_acts@" in listing.output and "scenes" in listing.output

    shown = runner.invoke(app, ["prompt", "render", "story_acts"])
    assert shown.exit_code == 0 and "你是电影剧情分析助手" in shown.output

    vars_file = tmp_path / "v.json"
    vars_file.write_text(json.dumps({"scenes": [{"id": "sc_001", "start": "00:00", "end": "01:00",
        "location": "雪山", "importance": 0.5, "summary": "出发"}]}))  # fmt: skip
    filled = runner.invoke(app, ["prompt", "render", "story_acts", "--vars", str(vars_file)])
    assert (
        filled.exit_code == 0 and "sc_001（00:00 – 01:00，地点：雪山，重要度 0.5）" in filled.output
    )
    assert "出发" in filled.output

    unknown = runner.invoke(app, ["prompt", "render", "nope"])
    assert unknown.exit_code == 1 and "unknown prompt" in unknown.output
    vars_file.write_text("[1, 2]")
    bad = runner.invoke(app, ["prompt", "render", "story_acts", "--vars", str(vars_file)])
    assert bad.exit_code == 1 and "must be a JSON object" in bad.output
    vars_file.write_text("{oops")
    bad2 = runner.invoke(app, ["prompt", "render", "story_acts", "--vars", str(vars_file)])
    assert bad2.exit_code == 1 and "not valid JSON" in bad2.output
    missing = runner.invoke(
        app, ["prompt", "render", "story_acts", "--vars", str(tmp_path / "none.json")]
    )
    assert missing.exit_code == 1
