"""The job use cases end to end: JobService enqueues, a Worker runs them through the real
Pipeline (fake LLM / TTS / shot detection, real ffmpeg, engine and stores)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from offscreen.config import AppConfig
from offscreen.domain.common import TimeRange
from offscreen.domain.job import Job
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.providers.ports import DetectionCanceled, LLMAuthError
from offscreen.services.jobs import (
    ANALYZE,
    BUILD_PLAN,
    GENERATE_SCRIPT,
    RENDER,
    JobService,
    options_from_scope,
    options_to_scope,
)
from offscreen.services.pipeline import Pipeline, Providers, RunOptions
from offscreen.stages.analysis.ingest import ingest
from offscreen.worker import Worker, WorkerSettings

OPTS = RunOptions(minutes=0.25)
FAST = WorkerSettings(poll_interval_s=0.02, heartbeat_timeout_s=1.0, progress_interval_s=0.0)


@pytest.fixture
def svc(cfg: AppConfig, fakes: Providers) -> Iterator[JobService]:
    s = JobService(cfg, providers=fakes)
    yield s
    s.close()


def worker(svc: JobService) -> Worker:
    return Worker(svc.jobs, svc.execute, svc.cfg.data_dir, FAST)


def asset_of(svc: JobService, movie: Path) -> str:
    return ingest(movie, svc.assets).asset.id


def status(svc: JobService, job: Job) -> str:
    got = svc.get(job.id)
    assert got is not None
    return got.status


def test_the_four_use_cases_take_a_film_to_a_video(svc: JobService, movie: Path) -> None:
    asset = asset_of(svc, movie)
    w = worker(svc)

    jobs = [
        svc.analyze(asset),
        svc.generate_script(asset, OPTS),
        svc.build_plan(asset, OPTS),
        svc.render(asset, OPTS),
    ]
    assert [j.stage for j in jobs] == [ANALYZE, GENERATE_SCRIPT, BUILD_PLAN, RENDER]
    assert [j.lane for j in jobs] == ["api", "api", "api", "cpu"]
    w.drain(timeout_s=120)

    done = [svc.get(j.id) for j in jobs]
    assert all(j is not None and j.status == "succeeded" and j.progress == 1.0 for j in done)
    final = next((svc.cfg.data_dir / "artifacts" / "output.render").glob("*/final.mp4"))
    assert final.stat().st_size > 0
    log = (svc.cfg.data_dir / str(jobs[0].log_path)).read_text(encoding="utf-8")
    assert "start analysis.story" in log


def test_a_script_job_analyzes_first_when_nothing_is_cached(svc: JobService, movie: Path) -> None:
    asset = asset_of(svc, movie)
    job = svc.generate_script(asset, OPTS)
    worker(svc).drain(timeout_s=120)
    assert status(svc, job) == "succeeded"
    assert list((svc.cfg.data_dir / "artifacts" / "analysis.story").iterdir())


def test_progress_covers_the_whole_chain_and_reports_stage_names(
    svc: JobService, movie: Path
) -> None:
    asset = asset_of(svc, movie)
    job = svc.analyze(asset)
    seen: list[tuple[float, str]] = []
    real = svc.jobs.report_progress

    def spy(job_id: str, attempt: int, frac: float, msg: str) -> None:
        seen.append((frac, msg))
        real(job_id, attempt, frac, msg)

    svc.jobs.report_progress = spy  # type: ignore[method-assign]
    worker(svc).drain(timeout_s=120)

    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs) and fracs[0] >= 0 and fracs[-1] <= 1.0
    assert any("analysis.shots" in m for _, m in seen) and any(
        "analysis.story" in m for _, m in seen
    )
    assert status(svc, job) == "succeeded"


def test_pressing_a_button_twice_does_not_queue_twice(svc: JobService, movie: Path) -> None:
    asset = asset_of(svc, movie)
    a = svc.generate_script(asset, OPTS)
    assert svc.generate_script(asset, OPTS).id == a.id
    assert svc.generate_script(asset, RunOptions(minutes=0.5)).id != a.id  # other options
    worker(svc).drain(timeout_s=120)
    assert svc.generate_script(asset, OPTS).id != a.id  # the first one is finished


def test_a_second_run_is_all_cache_hits(svc: JobService, movie: Path, fakes: Providers) -> None:
    asset = asset_of(svc, movie)
    w = worker(svc)
    svc.render(asset, OPTS)
    w.drain(timeout_s=120)
    calls = (len(fakes.llm.calls), len(fakes.tts.calls))  # type: ignore[attr-defined]

    again = svc.render(asset, OPTS)
    t0 = time.monotonic()
    w.drain(timeout_s=30)
    assert time.monotonic() - t0 < 5
    assert status(svc, again) == "succeeded"
    assert (len(fakes.llm.calls), len(fakes.tts.calls)) == calls  # type: ignore[attr-defined]


def test_analysis_takes_the_gpu_lane_only_when_a_local_asr_would_run(
    svc: JobService, movie: Path, tmp_path: Path
) -> None:
    with_subs = asset_of(svc, movie)
    assert svc.analyze(with_subs).lane == "api"  # external subtitles: no ASR

    bare = tmp_path / "bare" / "Bare.mp4"
    bare.parent.mkdir()
    bare.write_bytes(movie.read_bytes() + b"\0")  # another fingerprint, no .srt beside it
    assert svc.analyze(asset_of(svc, bare)).lane == "gpu"


def test_unknown_asset_and_bad_options_are_refused_up_front(svc: JobService) -> None:
    with pytest.raises(ValueError, match="unknown asset"):
        svc.analyze("ast_missing")
    with pytest.raises(ValueError, match="unknown asset"):
        svc.render("ast_missing")


def test_non_positive_length_is_refused(svc: JobService, movie: Path) -> None:
    asset = asset_of(svc, movie)
    with pytest.raises(ValueError, match="minutes"):
        svc.generate_script(asset, RunOptions(minutes=0))


class BrokenThenFine:
    """Delegates to a working LLM, except while `broken` is set."""

    def __init__(self, inner: FakeLLM) -> None:
        self.inner = inner
        self.broken = True

    def generate(self, *a: Any, **k: Any) -> Any:
        if self.broken:
            raise LLMAuthError("bad key")
        return self.inner.generate(*a, **k)


def test_failure_is_reported_on_the_job_and_manual_retry_succeeds(
    svc: JobService, movie: Path
) -> None:
    llm = BrokenThenFine(svc.providers.llm)  # type: ignore[arg-type]
    svc.providers.llm = llm  # type: ignore[assignment]
    asset = asset_of(svc, movie)
    job = svc.analyze(asset)
    w = worker(svc)
    w.drain(timeout_s=120)

    failed = svc.get(job.id)
    assert failed is not None and failed.status == "failed"
    assert failed.error == "LLMAuthError: bad key"
    assert failed.attempt == 1  # a person decides about retries, not the worker

    llm.broken = False
    svc.retry(job.id)
    w.drain(timeout_s=120)
    assert status(svc, job) == "succeeded"
    with pytest.raises(ValueError, match="only failed"):
        svc.retry(job.id)


class BlockingShots:
    """Shot detection that runs until it is told to stop."""

    id = "blocking-shots@1"

    def __init__(self) -> None:
        self.started = threading.Event()

    def detect(
        self, video: Path, *, on_progress: Any = None, should_cancel: Any = None
    ) -> list[TimeRange]:
        self.started.set()
        while not should_cancel():
            time.sleep(0.01)
        raise DetectionCanceled("shot detection canceled")


def test_canceling_a_running_job_stops_the_stage_and_leaves_no_artifact(
    svc: JobService, movie: Path
) -> None:
    blocking = BlockingShots()
    svc.providers.detector = blocking
    asset = asset_of(svc, movie)
    job = svc.analyze(asset)
    w = worker(svc)
    w.tick()
    assert blocking.started.wait(60)
    assert status(svc, job) == "running"

    svc.cancel(job.id)
    w.drain(timeout_s=30)
    got = svc.get(job.id)
    assert got is not None and got.status == "canceled" and got.error is None
    assert not (svc.cfg.data_dir / "artifacts" / "analysis.shots").exists() or not any(
        (svc.cfg.data_dir / "artifacts" / "analysis.shots").iterdir()
    )


def test_canceling_a_queued_job_means_it_never_runs(svc: JobService, movie: Path) -> None:
    job = svc.analyze(asset_of(svc, movie))
    assert svc.cancel(job.id).status == "canceled"
    worker(svc).drain(timeout_s=10)
    assert not (svc.cfg.data_dir / "artifacts").exists() or not list(
        (svc.cfg.data_dir / "artifacts").glob("analysis.*/*")
    )
    with pytest.raises(KeyError):
        svc.cancel("job_missing")


def test_options_round_trip_through_the_job_scope() -> None:
    opts = RunOptions(minutes=2.5, voice="v", style="funny", spoil_ending=False)
    assert options_from_scope(options_to_scope(opts)) == opts
    assert options_from_scope({}) == RunOptions()
    assert options_from_scope({"minutes": 1.0, "unknown": 1}) == RunOptions(minutes=1.0)


def test_stage_chain_lists_a_stage_with_its_upstream_first(
    cfg: AppConfig, fakes: Providers
) -> None:
    with Pipeline(cfg, fakes) as p:
        chain = p.stage_chain("creation.plan", "ast_x")
        assert chain[-1] == "creation.plan"
        assert (
            chain.index("analysis.proxy")
            < chain.index("analysis.shots")
            < chain.index("analysis.story")
        )
        assert len(chain) == len(set(chain))
        assert p.stage_chain("analysis.proxy", "ast_x") == ["analysis.proxy"]
        with pytest.raises(ValueError, match="unknown stage"):
            p.stage_chain("analysis.nope", "ast_x")
