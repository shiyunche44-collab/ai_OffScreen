"""The analysis report: what building an asset's MovieIndex cost, per stage and in total.

Time comes from `stage_runs` (every execution that was not a cache hit, failed ones included),
money and tokens from `llm_calls` attributed to the stage that made them. Counts are read from
the manifests of the cached artifacts, so the report runs nothing and needs no model key."""

from __future__ import annotations

from pydantic import BaseModel

from offscreen.config import AppConfig
from offscreen.domain.asset import MediaAsset
from offscreen.services.errors import NotFound
from offscreen.services.jobs import ANALYZE, JobService
from offscreen.services.pipeline import Pipeline
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo, LlmCallRepo, StageRunRepo

COUNTS: tuple[tuple[str, str, str], ...] = (
    ("shots", "analysis.shots", "shots"),
    ("transcript_lines", "analysis.transcript", "lines"),
    ("scenes", "analysis.scenes", "scenes"),
)
"""`(report field, stage, key in that stage's manifest meta)`."""


class LlmUsage(BaseModel):
    calls: int = 0
    failed_calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    """Subscription plans count as 0; the token counts are still recorded."""


class StageCost(BaseModel):
    stage: str
    cached: bool
    """The stage's output is built and up to date."""
    runs: int
    """Executions recorded, failed and canceled ones included (cache hits are not runs)."""
    failed_runs: int
    last_run_ms: int | None
    """Duration of the latest successful run."""
    total_ms: int
    """Time spent in all recorded runs: what it cost to get here, retries included."""
    llm: LlmUsage


class ReportCounts(BaseModel):
    """What the analysis found; None where the stage is not built (or does not exist yet)."""

    shots: int | None = None
    transcript_lines: int | None = None
    scenes: int | None = None
    characters: int | None = None


class AnalysisReport(BaseModel):
    asset: MediaAsset
    complete: bool
    """Every analysis stage is built and up to date."""
    stages: list[StageCost]
    """The analysis chain, upstream first."""
    counts: ReportCounts
    total_ms: int
    llm: LlmUsage


def _usage(rows: list) -> LlmUsage:  # type: ignore[type-arg]
    return LlmUsage(
        calls=len(rows),
        failed_calls=sum(r.status != "ok" for r in rows),
        in_tokens=sum(r.in_tokens for r in rows),
        out_tokens=sum(r.out_tokens for r in rows),
        cached_tokens=sum(r.cached_tokens for r in rows),
        cost_usd=sum(r.cost_usd for r in rows),
    )


class ReportService:
    def __init__(self, cfg: AppConfig, db: Database, jobs: JobService) -> None:
        self.cfg = cfg
        self.db = db
        self.jobs = jobs
        self.assets = AssetRepo(db)
        self.runs = StageRunRepo(db)
        self.calls = LlmCallRepo(db, cfg.data_dir)

    def analysis(self, asset_id: str) -> AnalysisReport:
        asset = self.assets.get(asset_id)
        if asset is None:
            raise NotFound(f"unknown asset {asset_id}")
        runs = self.runs.list(asset_id=asset_id)
        calls = self.calls.list(asset_id=asset_id)
        counts: dict[str, int | None] = {}
        with Pipeline(self.cfg, self.jobs.providers, db=self.db) as p:
            chain = p.stage_chain(ANALYZE, asset_id)
            stages: list[StageCost] = []
            for name in chain:
                mine = [r for r in runs if r.stage == name]
                good = [r for r in mine if r.status == "ok"]
                stages.append(
                    StageCost(
                        stage=name,
                        cached=p.peek(name, asset_id) is not None,
                        runs=len(mine),
                        failed_runs=len(mine) - len(good),
                        last_run_ms=good[-1].duration_ms if good else None,
                        total_ms=sum(r.duration_ms for r in mine),
                        llm=_usage([c for c in calls if c.stage == name]),
                    )
                )
            for field, stage, key in COUNTS:
                if stage in chain:
                    built = p.peek(stage, asset_id)
                    value = built.meta.get(key) if built is not None else None
                    counts[field] = value if isinstance(value, int) else None
        return AnalysisReport(
            asset=asset,
            complete=all(s.cached for s in stages),
            stages=stages,
            counts=ReportCounts(**counts),
            total_ms=sum(s.total_ms for s in stages),
            llm=_usage(calls),
        )


def _clock(ms: int) -> str:
    seconds = round(ms / 1000)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def format_report(r: AnalysisReport) -> str:
    """The report as a table for the terminal."""
    lines = [
        f"{r.asset.id}  {r.asset.title}  ({'complete' if r.complete else 'incomplete'})",
        "",
        f"{'stage':<22}{'built':<7}{'runs':>5}{'last':>9}{'total':>9}{'calls':>7}"
        f"{'in tok':>10}{'out tok':>9}{'cost $':>9}",
    ]
    for s in r.stages:
        last = _clock(s.last_run_ms) if s.last_run_ms is not None else "-"
        runs = f"{s.runs}" + (f" ({s.failed_runs} failed)" if s.failed_runs else "")
        lines.append(
            f"{s.stage:<22}{'yes' if s.cached else 'no':<7}{runs:>5}{last:>9}"
            f"{_clock(s.total_ms):>9}{s.llm.calls:>7}{s.llm.in_tokens:>10}{s.llm.out_tokens:>9}"
            f"{s.llm.cost_usd:>9.4f}"
        )
    u = r.llm
    lines += [
        "",
        f"total  {_clock(r.total_ms)}   {u.calls} model calls ({u.failed_calls} failed)   "
        f"{u.in_tokens} in / {u.out_tokens} out tokens ({u.cached_tokens} cached)   "
        f"${u.cost_usd:.4f}",
    ]
    found = [
        f"{label} {value}"
        for label, value in (
            ("shots", r.counts.shots),
            ("transcript lines", r.counts.transcript_lines),
            ("scenes", r.counts.scenes),
            ("characters", r.counts.characters),
        )
        if value is not None
    ]
    lines.append("found  " + (", ".join(found) if found else "nothing built yet"))
    return "\n".join(lines)
