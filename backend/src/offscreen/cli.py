"""Typer CLI (thin layer: calls services only)."""

import json
from pathlib import Path
from typing import Annotated

import typer

from offscreen.config import ConfigError, load_config
from offscreen.domain.index import SelectionLabel
from offscreen.server import default_web_dir, run_worker, serve
from offscreen.services.app import AppServices
from offscreen.services.config_view import show_config
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.pipeline import (
    DEFAULT_STYLE,
    EXPECTED_ERRORS,
    Pipeline,
    RunOptions,
)
from offscreen.services.prompts import list_prompts, render_prompt
from offscreen.services.report import format_report
from offscreen.services.styles import format_style, get_style, list_styles

app = typer.Typer(help="AI OffScreen: movie commentary pipeline", no_args_is_help=True)
config_app = typer.Typer(help="Configuration commands", no_args_is_help=True)
app.add_typer(config_app, name="config")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Path to config.yaml")]
MinutesOpt = Annotated[float, typer.Option(help="Target length of the commentary, in minutes")]
VoiceOpt = Annotated[str | None, typer.Option(help="Voice id (default: tts.default_voice)")]
StyleOpt = Annotated[str, typer.Option(help="Writing style preset (see: offscreen style list)")]
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


@app.command("report")
def report_command(
    asset: Annotated[str, typer.Argument(help="Asset id (ast_…)")],
    config: ConfigOpt = None,
) -> None:
    """What analyzing a movie cost: time and model usage per stage, and what it found."""
    try:
        cfg = load_config(config)
        with AppServices(cfg) as services:
            typer.echo(format_report(services.report.analysis(asset)))
    except (ConfigError, NotFound) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


prompt_app = typer.Typer(help="The prompt templates models are given", no_args_is_help=True)
app.add_typer(prompt_app, name="prompt")


@prompt_app.command("list")
def prompt_list() -> None:
    """Every template with its version and the variables it reads."""
    for _name, version, names in list_prompts():
        typer.echo(f"{version:<28} {', '.join(names) or '-'}")


@prompt_app.command("render")
def prompt_render(
    name: Annotated[str, typer.Argument(help="Template name, e.g. story_acts")],
    vars_: Annotated[
        Path | None,
        typer.Option("--vars", help="JSON file with the variables (missing ones show as {{ x }})"),
    ] = None,
) -> None:
    """Print a template as the model would get it."""
    try:
        text = vars_.read_text(encoding="utf-8") if vars_ else None
        version, body = render_prompt(name, text)
    except (OSError, NotFound, InvalidInput) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"# {version}", err=True)
    typer.echo(body)


style_app = typer.Typer(help="The writing style presets", no_args_is_help=True)
app.add_typer(style_app, name="style")


@style_app.command("list")
def style_list() -> None:
    """Every preset: id, name and who it suits."""
    for preset in list_styles():
        typer.echo(f"{preset.id:<12} {preset.name:<10} {preset.description}")


@style_app.command("show")
def style_show(style_id: Annotated[str, typer.Argument(help="Preset id, e.g. suspense")]) -> None:
    """Print a preset in full: tone, structure, hooks, phrases, banned words."""
    try:
        typer.echo(format_style(get_style(style_id)))
    except NotFound as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


cuts_app = typer.Typer(
    help="Hand-marked shot cuts and how detection measures up", no_args_is_help=True
)
app.add_typer(cuts_app, name="cuts")


@cuts_app.command("evaluate")
def cuts_evaluate(
    asset: Annotated[str, typer.Argument(help="Asset id (ast_…)")],
    tolerance: Annotated[int, typer.Option(help="Frames of slack for a match")] = 2,
    raw: Annotated[
        bool, typer.Option("--raw", help="Run the detector itself (slow) instead of the shots")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Precision / recall / F1 of detected cuts against the marked ones."""
    try:
        cfg = load_config(config)
        with AppServices(cfg) as services:
            r = services.annotations.evaluate_cuts(
                asset, tolerance=tolerance, source="detector" if raw else "shots"
            )
    except (ConfigError, NotFound, InvalidInput) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"{r.asset_id}  {r.source}  tolerance ±{r.tolerance} frames")
    typer.echo(f"marked {r.marked}   detected {r.detected}   matched {r.true_positives}")
    typer.echo(f"precision {r.precision:.3f}   recall {r.recall:.3f}   F1 {r.f1:.3f}")
    typer.echo(f"false positives (frames): {', '.join(map(str, r.false_positives)) or '-'}")
    typer.echo(f"missed (frames):          {', '.join(map(str, r.false_negatives)) or '-'}")


@cuts_app.command("import")
def cuts_import(
    asset: Annotated[str, typer.Argument(help="Asset id (ast_…)")],
    file: Annotated[Path, typer.Argument(help='JSON: a list of frames, or {"cuts": [...]}')],
    config: ConfigOpt = None,
) -> None:
    """Save marked cuts from a JSON file (frame numbers, counted at the movie's frame rate)."""
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
        frames = raw["cuts"] if isinstance(raw, dict) else raw
        if not isinstance(frames, list) or not all(isinstance(f, int) for f in frames):
            raise ValueError("expected a list of frame numbers")
        cfg = load_config(config)
        with AppServices(cfg) as services:
            view = services.annotations.save_cuts(asset, frames)
    except (OSError, ValueError, KeyError, ConfigError, NotFound, InvalidInput) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"saved {len(view.cuts)} cuts for {asset}")


@cuts_app.command("export")
def cuts_export(
    asset: Annotated[str, typer.Argument(help="Asset id (ast_…)")],
    config: ConfigOpt = None,
) -> None:
    """Print the marked cuts as JSON."""
    try:
        cfg = load_config(config)
        with AppServices(cfg) as services:
            view = services.annotations.cuts(asset)
    except (ConfigError, NotFound) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        json.dumps(
            {"asset_id": view.asset_id, "fps": [view.fps_num, view.fps_den], "cuts": view.cuts}
        )
    )


select_app = typer.Typer(
    help="Hand-labelled footage choices and how well the ranking of shots finds them",
    no_args_is_help=True,
)
app.add_typer(select_app, name="select")


@select_app.command("evaluate")
def select_evaluate(
    asset: Annotated[str, typer.Argument(help="Asset id (ast_…)")],
    k: Annotated[int, typer.Option("-k", help="How many top shots count as found")] = 5,
    config: ConfigOpt = None,
) -> None:
    """First-choice rate and top-k recall of the shot ranking against the labelled choices."""
    try:
        cfg = load_config(config)
        with AppServices(cfg) as services:
            r = services.annotations.evaluate_selection(asset, k=k)
    except (ConfigError, NotFound, InvalidInput) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    search = "with" if r.vector_search else "without"
    typer.echo(f"{r.asset_id}  {r.labels} labels, ranked {search} the vector search")
    typer.echo(f"first choice acceptable  {r.first_choice_rate:.3f}")
    typer.echo(f"acceptable shot in top {r.k}  {r.top_k_hit_rate:.3f}")
    typer.echo(f"top-{r.k} recall           {r.top_k_recall:.3f}")
    typer.echo(f"mean reciprocal rank     {r.mean_reciprocal_rank:.3f}")
    missed = [x for x in r.results if not x.top_k_hit]
    typer.echo(f"not in the top {r.k}: {', '.join(x.label_id for x in missed) or '-'}")
    wrong = [x for x in r.results if x.first_rank != 1 and x.top_k_hit]
    late = ", ".join(f"{x.label_id}(#{x.first_rank})" for x in wrong) or "-"
    typer.echo(f"found, but not first:  {late}")


@select_app.command("import")
def select_import(
    asset: Annotated[str, typer.Argument(help="Asset id (ast_…)")],
    file: Annotated[
        Path,
        typer.Argument(
            help='JSON: a list of labels, or {"labels": [...]}; a label is '
            '{"id", "text", "scene_refs": [...], "acceptable": [shot ids], "note"?}'
        ),
    ],
    config: ConfigOpt = None,
) -> None:
    """Save labelled footage choices from a JSON file (replaces the earlier ones)."""
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
        items = raw["labels"] if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            raise ValueError("expected a list of labels")
        labels = [SelectionLabel.model_validate(x) for x in items]
        cfg = load_config(config)
        with AppServices(cfg) as services:
            saved = services.annotations.save_selection(asset, labels)
    except (OSError, ValueError, KeyError, ConfigError, NotFound, InvalidInput) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"saved {len(saved)} labels for {asset}")


@select_app.command("export")
def select_export(
    asset: Annotated[str, typer.Argument(help="Asset id (ast_…)")],
    config: ConfigOpt = None,
) -> None:
    """Print the labelled footage choices as JSON."""
    try:
        cfg = load_config(config)
        with AppServices(cfg) as services:
            labels = services.annotations.selection(asset)
    except (ConfigError, NotFound) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        json.dumps(
            {"asset_id": asset, "labels": [x.model_dump(mode="json") for x in labels]},
            ensure_ascii=False,
        )
    )


@app.command("serve")
def serve_command(
    host: Annotated[
        str, typer.Option(help="Address to listen on (no authentication: keep it local)")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on")] = 8000,
    web_dir: Annotated[
        Path | None,
        typer.Option("--web-dir", help="Built front end (default: frontend/dist if built)"),
    ] = None,
    no_worker: Annotated[
        bool, typer.Option("--no-worker", help="API only; run `offscreen worker` elsewhere")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Start the web app: API, front end and job worker in one process."""
    try:
        cfg = load_config(config)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    dist = web_dir or default_web_dir()
    if dist is None:
        typer.echo(
            "note: no built front end (run `npm run build` in frontend/); serving the API only",
            err=True,
        )
    typer.echo(f"offscreen: http://{host}:{port}", err=True)
    serve(cfg, host=host, port=port, web_dir=dist, with_worker=not no_worker)


@app.command("worker")
def worker_command(config: ConfigOpt = None) -> None:
    """Run the job worker alone (for use with `serve --no-worker`)."""
    try:
        cfg = load_config(config)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    run_worker(cfg)


if __name__ == "__main__":
    app()
