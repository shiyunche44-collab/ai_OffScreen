"""Application configuration (config.yaml + environment variables).

Secrets are never stored in config: each provider names the environment variable
that holds its key (`api_key_env`) and the key is read lazily at call time."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LOCAL_ENGINES = frozenset({"faster_whisper", "funasr", "edge_tts", "cosyvoice"})
# Volcengine Ark *Coding Plan* keys are restricted to AI coding tools (see docs/PROVIDERS.md §4).
FORBIDDEN_KEY_ENVS = frozenset({"ARK_API_KEY"})
FORBIDDEN_URL_PARTS = ("/api/coding",)


class ConfigError(ValueError):
    pass


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderCfg(_Cfg):
    kind: Literal["openai_compat", "volc_tts"]
    base_url: str | None = None
    api_key_env: str
    max_concurrency: int = Field(default=2, ge=1, le=64)
    json_mode: Literal["native", "prompt"] = "prompt"
    extra_body: dict[str, Any] = {}
    """Provider-specific fields merged into every chat request (e.g. MiniMax `reasoning_split`)."""
    enabled: bool = True

    @model_validator(mode="after")
    def _safety(self) -> Self:
        if self.api_key_env in FORBIDDEN_KEY_ENVS:
            raise ValueError(
                f"{self.api_key_env} is a Coding Plan key restricted to AI coding tools; "
                "use a pay-as-you-go key in ARK_PAYG_API_KEY instead (docs/PROVIDERS.md §4)"
            )
        if self.base_url and any(p in self.base_url for p in FORBIDDEN_URL_PARTS):
            raise ValueError(f"base_url {self.base_url!r} targets a coding-plan endpoint")
        if self.kind == "openai_compat" and not self.base_url:
            raise ValueError("openai_compat provider needs base_url")
        return self


class TaskCfg(_Cfg):
    provider: str
    model: str
    shots_per_request: int | None = Field(default=None, ge=1, le=64)


class AsrCfg(_Cfg):
    provider: str = "faster_whisper"
    model: str = "large-v3"
    device: Literal["cuda", "cpu"] = "cuda"


class TtsCfg(_Cfg):
    provider: str = "minimax"
    model: str = "speech-2.8-hd"
    default_voice: str = "male-qn-qingse"
    sample_rate: int = 32000


class BudgetCfg(_Cfg):
    confirm_above_tokens: int = Field(default=2_000_000, ge=0)


class AppConfig(_Cfg):
    data_dir: Path = Path("./data")
    media_roots: list[Path] = []
    providers: dict[str, ProviderCfg] = {}
    tasks: dict[str, TaskCfg] = {}
    asr: AsrCfg = AsrCfg()
    tts: TtsCfg = TtsCfg()
    budget: BudgetCfg = BudgetCfg()

    @model_validator(mode="after")
    def _references(self) -> Self:
        for name, t in self.tasks.items():
            p = self.providers.get(t.provider)
            if p is None:
                raise ValueError(f"task {name!r} uses undefined provider {t.provider!r}")
            if not p.enabled:
                raise ValueError(f"task {name!r} uses disabled provider {t.provider!r}")
        for label, prov in (("asr", self.asr.provider), ("tts", self.tts.provider)):
            if prov in LOCAL_ENGINES:
                continue
            p = self.providers.get(prov)
            if p is None or not p.enabled:
                raise ValueError(
                    f"{label}.provider {prov!r} is not a local engine or enabled provider"
                )
        return self

    def api_key(self, provider: str) -> str:
        """Read a provider's key from the environment at call time."""
        cfg = self.providers[provider]
        value = os.environ.get(cfg.api_key_env, "").strip()
        if not value:
            raise ConfigError(
                f"environment variable {cfg.api_key_env} is not set (provider {provider!r})"
            )
        return value


def load_config(path: Path | None = None) -> AppConfig:
    """Load config from `path`, $OFFSCREEN_CONFIG, or ./config.yaml (in that order)."""
    chosen = path or (Path(p) if (p := os.environ.get("OFFSCREEN_CONFIG")) else Path("config.yaml"))
    if not chosen.exists():
        raise ConfigError(f"config file not found: {chosen} (copy config.example.yaml)")
    raw = yaml.safe_load(chosen.read_text(encoding="utf-8")) or {}
    try:
        return AppConfig.model_validate(raw)
    except ValueError as e:
        raise ConfigError(f"invalid config {chosen}: {e}") from e


def redacted(cfg: AppConfig) -> dict[str, object]:
    """Config as plain data with key *values* never included: only whether each
    provider's variable is set."""
    data = cfg.model_dump(mode="json")
    for name, p in data["providers"].items():
        p["api_key_status"] = "set" if os.environ.get(p["api_key_env"], "").strip() else "MISSING"
        data["providers"][name] = p
    return data
