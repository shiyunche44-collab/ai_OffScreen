from typer.testing import CliRunner

from offscreen.cli import app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AI OffScreen" in result.output
