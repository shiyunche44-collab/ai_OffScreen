"""Typer CLI (thin layer: calls services only)."""

from pathlib import Path
from typing import Annotated

import typer

from offscreen.config import ConfigError, load_config
from offscreen.services.config_view import show_config
from offscreen.services.pipeline import (
    DEFAULT_STYLE,
    EXPECTED_ERRORS,
    Pipeline,
    RunOptions,
)

app = typer.Typer(help="AI OffScreen: movie commentary pipeline", no_args_is_help=True)
config_app = typer.Typer(help="Configuration commands", no_args_is_help=True)
app.add_typer(config_app, name="config")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Path to config.yaml")]
MinutesOpt = Annotated[float, typer.Option(help="Target length of the commentary, in minutes")]
VoiceOpt = Annotated[str | None, typer.Option(help="Voice id (default: tts.default_voice)")]
StyleOpt = Annotated[str, typer.Option(help="Writing style hint for the script")]
NoSpoilOpt = Annotated[bool, typer.Option("--no-spoilers", help="Keep the ending out of it")]


@config_app.command("show")
def config_show(path: ConfigOpt = None) -> None:
    """Print the effective config; secret values are never shown."""
    try:
        typer.echo(show_config(path))
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


class _Progress:
    """Prints a stage's progress on stderr, at most every 10 %."""

    def __init__(self) -> None:
        self._last: dict[str, int] = {}

    def __call__(self, stage: str, frac: float, msg: str) -> None:
        step = int(frac * 10)
        if self._last.get(stage) != step or frac >= 1.0:
            self._last[stage] = step
            typer.echo(f"  {stage:<20} {frac:4.0%}  {msg}", err=True)


def _execute(config: Path | None, stage: str | None, target: str, opts: RunOptions) -> None:
    try:
        cfg = load_config(config)
        with Pipeline(
            cfg,
            progress=_Progress(),
            on_resolved=lambda s, hit: typer.echo(
                f"{'cached' if hit else 'done  '}  {s}", err=True
            ),
        ) as pipeline:
            if stage is None:
                result = pipeline.run_all(Path(target), opts)
            else:
                result = pipeline.run_stage(stage, target, opts)
    except EXPECTED_ERRORS as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    ran = sum(not r.cached for r in result.stages)
    typer.echo(f"asset     {result.asset_id}  ({result.title})")
    typer.echo(f"artifact  {result.artifact.dir}")
    typer.echo(f"stages    {len(result.stages)} ({ran} run, {len(result.stages) - ran} cached)")
    if stage is None:
        typer.echo(f"video     {result.final}")


@app.command("run-all")
def run_all(
    movie: Annotated[Path, typer.Argument(help="Movie file")],
    minutes: MinutesOpt = 3.0,
    voice: VoiceOpt = None,
    style: StyleOpt = DEFAULT_STYLE,
    no_spoilers: NoSpoilOpt = False,
    config: ConfigOpt = None,
) -> None:
    """Movie file -> commentary video, running whatever is not cached yet."""
    _execute(config, None, str(movie), RunOptions(minutes, voice, style, not no_spoilers))


@app.command("stage")
def stage(
    name: Annotated[str, typer.Argument(help="Stage name, e.g. analysis.shots")],
    asset: Annotated[str, typer.Option(help="Asset id (ast_…) or a movie file to import")],
    minutes: MinutesOpt = 3.0,
    voice: VoiceOpt = None,
    style: StyleOpt = DEFAULT_STYLE,
    no_spoilers: NoSpoilOpt = False,
    config: ConfigOpt = None,
) -> None:
    """Run one stage (and whatever it depends on) for an asset."""
    _execute(config, name, asset, RunOptions(minutes, voice, style, not no_spoilers))


if __name__ == "__main__":
    app()
