from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from offscreen.cli import app
from offscreen.config import AppConfig
from offscreen.server import default_web_dir, worker_settings

runner = CliRunner()


def config(**extra: object) -> AppConfig:
    return AppConfig.model_validate({"tts": {"provider": "edge_tts"}, **extra})


def test_worker_settings_follow_the_config_and_gpu_stays_at_one() -> None:
    s = worker_settings(
        config(worker={"cpu_concurrency": 3, "api_concurrency": 8, "heartbeat_timeout_s": 12})
    )
    assert dict(s.lane_limits) == {"gpu": 1, "cpu": 3, "api": 8}
    assert s.heartbeat_timeout_s == 12
    assert dict(worker_settings(config()).lane_limits) == {"gpu": 1, "cpu": 2, "api": 4}


@pytest.mark.parametrize(
    "bad",
    [
        {"cpu_concurrency": 0},
        {"api_concurrency": 0},
        {"heartbeat_timeout_s": 1},
        {"gpu_concurrency": 2},
    ],
)
def test_worker_config_is_validated(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        config(worker=bad)


def test_the_example_config_has_a_worker_section() -> None:
    import yaml

    example = Path(__file__).resolve().parents[3] / "config.example.yaml"
    raw = yaml.safe_load(example.read_text(encoding="utf-8"))
    assert AppConfig.model_validate(raw).worker.api_concurrency == 4


def test_serve_and_worker_are_listed_and_documented() -> None:
    out = runner.invoke(app, ["--help"]).output
    assert "serve" in out and "worker" in out
    help_text = runner.invoke(app, ["serve", "--help"]).output
    for option in ("--host", "--port", "--web-dir", "--no-worker", "--config"):
        assert option in help_text


def test_serve_and_worker_report_a_missing_config_instead_of_a_traceback(tmp_path: Path) -> None:
    missing = str(tmp_path / "nope.yaml")
    for command in ("serve", "worker"):
        r = runner.invoke(app, [command, "--config", missing])
        assert r.exit_code == 1 and "config file not found" in r.output, command


def test_default_web_dir_is_only_reported_when_it_is_built() -> None:
    found = default_web_dir()
    assert found is None or (found / "index.html").is_file()
