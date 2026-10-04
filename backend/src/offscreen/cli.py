"""Typer CLI (thin layer: calls services only)."""

from pathlib import Path
from typing import Annotated

import typer

from offscreen.config import ConfigError
from offscreen.services.config_view import show_config

app = typer.Typer(help="AI OffScreen: movie commentary pipeline", no_args_is_help=True)
config_app = typer.Typer(help="Configuration commands", no_args_is_help=True)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    path: Annotated[Path | None, typer.Option("--config", "-c", help="Path to config.yaml")] = None,
) -> None:
    """Print the effective config; secret values are never shown."""
    try:
        typer.echo(show_config(path))
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
