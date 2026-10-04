from pathlib import Path

import pytest
from typer.testing import CliRunner

from offscreen.cli import app
from offscreen.config import AppConfig, ConfigError, load_config, redacted

REPO = Path(__file__).resolve().parents[3]
EXAMPLE = REPO / "config.example.yaml"


def base() -> dict:  # type: ignore[type-arg]
    return {
        "providers": {
            "p": {"kind": "openai_compat", "base_url": "https://x/v1", "api_key_env": "P_KEY"}
        },
        "tasks": {"story": {"provider": "p", "model": "m"}},
        "tts": {"provider": "edge_tts"},
    }


def test_example_config_loads() -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.providers["minimax"].max_concurrency == 2
    assert cfg.tasks["shot_caption"].shots_per_request == 8
    assert not cfg.providers["ark"].enabled


def test_coding_plan_endpoint_is_rejected() -> None:
    d = base()
    d["providers"]["p"]["base_url"] = "https://ark.cn-beijing.volces.com/api/coding/v3"
    with pytest.raises(ValueError, match="coding-plan"):
        AppConfig.model_validate(d)


def test_coding_plan_key_variable_is_rejected() -> None:
    d = base()
    d["providers"]["p"]["api_key_env"] = "ARK_API_KEY"
    with pytest.raises(ValueError, match="Coding Plan"):
        AppConfig.model_validate(d)


def test_task_must_use_defined_enabled_provider() -> None:
    d = base()
    d["tasks"]["story"]["provider"] = "nope"
    with pytest.raises(ValueError, match="undefined provider"):
        AppConfig.model_validate(d)
    d = base()
    d["providers"]["p"]["enabled"] = False
    with pytest.raises(ValueError, match="disabled provider"):
        AppConfig.model_validate(d)


def test_unknown_keys_are_rejected() -> None:
    d = base()
    d["surprise"] = 1
    with pytest.raises(ValueError):
        AppConfig.model_validate(d)


def test_api_key_read_lazily_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig.model_validate(base())
    monkeypatch.delenv("P_KEY", raising=False)
    with pytest.raises(ConfigError, match="P_KEY"):
        cfg.api_key("p")
    monkeypatch.setenv("P_KEY", "  secret-value  ")
    assert cfg.api_key("p") == "secret-value"


def test_redaction_never_contains_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P_KEY", "super-secret-123")
    out = redacted(AppConfig.model_validate(base()))
    assert "super-secret-123" not in str(out)
    assert out["providers"]["p"]["api_key_status"] == "set"  # type: ignore[index]
    monkeypatch.delenv("P_KEY")
    assert (
        redacted(AppConfig.model_validate(base()))["providers"]["p"]["api_key_status"] == "MISSING"
    )  # type: ignore[index]


def test_cli_config_show(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "super-secret-456")
    r = CliRunner().invoke(app, ["config", "show", "--config", str(EXAMPLE)])
    assert r.exit_code == 0, r.output
    assert "super-secret-456" not in r.output
    assert "MINIMAX_API_KEY" in r.output and "api_key_status: set" in r.output


def test_cli_missing_config_is_a_clean_error(tmp_path: Path) -> None:
    r = CliRunner().invoke(app, ["config", "show", "--config", str(tmp_path / "nope.yaml")])
    assert r.exit_code == 1 and "not found" in r.output
