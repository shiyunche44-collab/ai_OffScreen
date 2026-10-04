"""Typer CLI (thin layer: calls services only)."""

import typer

app = typer.Typer(help="AI OffScreen: movie commentary pipeline", no_args_is_help=True)


@app.callback()
def main() -> None:
    """AI OffScreen command line."""


if __name__ == "__main__":
    app()
