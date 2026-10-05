"""Wiring: the configured LLM with call accounting going to the `llm_calls` table."""

from __future__ import annotations

from collections.abc import Iterable

from offscreen.config import AppConfig
from offscreen.domain.llm import LlmCallRecord
from offscreen.log import current_job_id, current_stage
from offscreen.providers.adapters.openai_compat import OpenAICompatLLM
from offscreen.providers.ports import Recorder
from offscreen.store.db import Database
from offscreen.store.repos import LlmCallRepo


def call_recorder(cfg: AppConfig, db: Database) -> Recorder:
    """Writes each model call to `llm_calls`. Stages do not pass their context down to the
    call; the engine and the worker bind it (`log.bind_stage` / `bind_job`), and it is attached
    here so the call can be attributed to a stage, an asset and a job."""
    calls = LlmCallRepo(db, cfg.data_dir)

    def record(rec: LlmCallRecord) -> None:
        running = current_stage()
        if running is not None and rec.stage is None:
            rec = rec.model_copy(update={"stage": running[0], "asset_id": running[1]})
        if rec.job_id is None:
            rec = rec.model_copy(update={"job_id": current_job_id()})
        calls.add(rec)

    return record


def build_llm(cfg: AppConfig, db: Database) -> OpenAICompatLLM:
    return OpenAICompatLLM(cfg, call_recorder(cfg, db))


def task_models(cfg: AppConfig, tasks: Iterable[str]) -> dict[str, str]:
    """`task -> "provider/model"` as configured; part of the cache key of stages using them."""
    return {t: f"{cfg.tasks[t].provider}/{cfg.tasks[t].model}" for t in tasks if t in cfg.tasks}
