"""Real calls to the configured providers: one tiny request each (`pytest -m heavy`).
Skipped when the key is not in the environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

from offscreen.config import AppConfig
from offscreen.domain.llm import LlmCallRecord
from offscreen.providers.adapters.openai_compat import OpenAICompatLLM
from offscreen.providers.ports import Message

pytestmark = pytest.mark.heavy

EXAMPLE = Path(__file__).resolve().parents[3] / "config.example.yaml"


class Sum(BaseModel):
    answer: int


def real_llm(
    provider: str, model: str, tmp_path: Path
) -> tuple[OpenAICompatLLM, list[LlmCallRecord]]:
    raw = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    raw["data_dir"] = str(tmp_path)
    raw["tasks"] = {"ping": {"provider": provider, "model": model}}
    cfg = AppConfig.model_validate(raw)
    if not os.environ.get(cfg.providers[provider].api_key_env, "").strip():
        pytest.skip(f"{cfg.providers[provider].api_key_env} is not set")
    records: list[LlmCallRecord] = []
    return OpenAICompatLLM(cfg, records.append), records


@pytest.mark.parametrize(
    ("provider", "model"), [("minimax", "MiniMax-M3"), ("deepseek", "deepseek-flash")]
)
def test_one_real_structured_call(provider: str, model: str, tmp_path: Path) -> None:
    llm, records = real_llm(provider, model, tmp_path)
    out = llm.generate(
        "ping",
        [Message("user", "What is 17 + 25? Answer with the number only, in the schema.")],
        Sum,
        prompt_version="heavy@1",
        max_tokens=1024,
    )
    assert out.answer == 42
    rec = records[-1]
    assert rec.status == "ok" and rec.in_tokens > 0 and rec.out_tokens > 0
    print(provider, rec.in_tokens, rec.out_tokens, rec.cached_tokens, rec.latency_ms, "ms",
          "reasoning chars:", len(rec.response.get("reasoning_content") or ""))  # fmt: skip
