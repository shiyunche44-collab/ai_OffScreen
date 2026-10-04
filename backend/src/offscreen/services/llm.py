"""Wiring: the configured LLM with call accounting going to the `llm_calls` table."""

from __future__ import annotations

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
