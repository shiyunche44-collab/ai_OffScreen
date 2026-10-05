"""Accounting record for one LLM request (the `llm_calls` table, ARCHITECTURE §8.3)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from offscreen.domain.common import Strict


class LlmCallRecord(Strict):
    """Bodies exclude credentials and image bytes."""

    task: str
    provider: str
    model: str
    prompt_version: str
    status: Literal["ok", "error"]
    job_id: str | None = None
    stage: str | None = None
    asset_id: str | None = None
    error: str | None = None
    retries: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0  # subscription plans count as 0; tokens are still recorded
    latency_ms: int = 0
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] = Field(default_factory=dict)
    """Includes `reasoning_content` when the model returned any; it is never parsed."""
