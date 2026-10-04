"""Wiring: the configured LLM with call accounting going to the `llm_calls` table."""

from __future__ import annotations

from collections.abc import Iterable

from offscreen.config import AppConfig
from offscreen.domain.llm import LlmCallRecord
from offscreen.providers.adapters.openai_compat import OpenAICompatLLM
from offscreen.store.db import Database
from offscreen.store.repos import LlmCallRepo


def build_llm(cfg: AppConfig, db: Database) -> OpenAICompatLLM:
    calls = LlmCallRepo(db, cfg.data_dir)

    def record(rec: LlmCallRecord) -> None:
        calls.add(rec)

    return OpenAICompatLLM(cfg, record)


def task_models(cfg: AppConfig, tasks: Iterable[str]) -> dict[str, str]:
    """`task -> "provider/model"` as configured; part of the cache key of stages using them."""
    return {t: f"{cfg.tasks[t].provider}/{cfg.tasks[t].model}" for t in tasks if t in cfg.tasks}
