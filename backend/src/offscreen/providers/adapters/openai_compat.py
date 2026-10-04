"""LLM over any OpenAI-compatible `/chat/completions` endpoint (MiniMax, DeepSeek, Ark, vLLM…).

Quirks are documented in docs/PROVIDERS.md. The structured-output contract (ARCHITECTURE
§8.3): `generate()` returns a schema-validated object or raises; in "prompt" mode the schema
goes into the system prompt, the JSON is extracted from the reply, and one repair round trip
is allowed when validation fails. Reasoning text is recorded but never parsed."""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from offscreen.algo.jsonreply import NoJsonFound, extract_json
from offscreen.config import AppConfig, ProviderCfg
from offscreen.domain.llm import LlmCallRecord
from offscreen.providers.ports import (
    LLMAuthError,
    LLMError,
    LLMQuotaExhausted,
    LLMRateLimited,
    LLMSchemaError,
    M,
    Message,
    Recorder,
)

MAX_RETRIES = 3  # transient failures only; so up to 4 requests per chat
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0
TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=60.0)

FailureKind = Literal["quota", "auth", "transient", "fatal"]

# MiniMax reports errors in `base_resp`, sometimes with HTTP 200.
_MINIMAX_QUOTA = {1008, 2056}  # insufficient balance; token-plan usage window exhausted
_MINIMAX_AUTH = {1004, 2049}
_MINIMAX_TRANSIENT = {1000, 1001, 1002, 1013, 1033, 1039, 1041}  # timeout, RPM/TPM, internal
# Deliberately excludes plain "rate limit", which is transient.
_QUOTA_WORDS = re.compile(
    r"insufficient|balance|quota|usage[ _]limit|exhaust|billing|额度|余额|用量", re.IGNORECASE
)


def classify_failure(status: int, body: Any, text: str = "") -> FailureKind:
    """Decide how to react to an error response. `body` is the parsed JSON, if any."""
    base = body.get("base_resp") if isinstance(body, dict) else None
    code = base.get("status_code") if isinstance(base, dict) else None
    if isinstance(code, int) and code != 0:
        if code in _MINIMAX_QUOTA:
            return "quota"
        if code in _MINIMAX_AUTH:
            return "auth"
        if code in _MINIMAX_TRANSIENT:
            return "transient"
    if status in (401, 403):
        return "auth"
    if status == 402:  # DeepSeek: Insufficient Balance
        return "quota"
    if status == 429:
        return "quota" if _QUOTA_WORDS.search(text) else "transient"
    if status in (408, 409, 425) or status >= 500:
        return "transient"
    if isinstance(code, int) and code != 0:
        return "fatal"
    return "fatal"


def usage_of(usage: dict[str, Any]) -> tuple[int, int, int]:
    """(input, output, cached input) tokens; providers name the cached count differently."""
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(cached),
    )


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning: str
    finish_reason: str | None
    in_tokens: int
    out_tokens: int
    cached_tokens: int
    retries: int
    response: dict[str, Any]


def _wire_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
            parts += [{"type": "image_url", "image_url": {"url": u}} for u in m.images]
            out.append({"role": m.role, "content": parts})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _redact_images(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy of the request for the log: image bytes are replaced by a size and hash."""
    msgs = []
    for m in payload["messages"]:
        content = m["content"]
        if isinstance(content, list):
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"<image {len(p['image_url']['url'])} chars "
                        f"sha256:{hashlib.sha256(p['image_url']['url'].encode()).hexdigest()[:16]}>"
                    },
                }
                if p.get("type") == "image_url"
                else p
                for p in content
            ]
        msgs.append({**m, "content": content})
    return {**payload, "messages": msgs}


class ProviderClient:
    """One provider's HTTP side: auth, concurrency limit, backoff, error classification."""

    def __init__(
        self,
        name: str,
        cfg: ProviderCfg,
        api_key: Callable[[], str],
        *,
        limiter: threading.Semaphore,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        assert cfg.base_url
        self.name = name
        self.cfg = cfg
        self._api_key = api_key
        self._limiter = limiter
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=cfg.base_url.rstrip("/"), timeout=TIMEOUT, transport=transport
        )

    def close(self) -> None:
        self._http.close()

    def chat(self, payload: dict[str, Any]) -> ChatResult:
        """POST one chat request, retrying transient failures. Raises LLMQuotaExhausted,
        LLMAuthError, LLMRateLimited (retries used up) or LLMError."""
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        last = ""
        for attempt in range(MAX_RETRIES + 1):
            retry_after: float | None = None
            with self._limiter:
                try:
                    resp = self._http.post("/chat/completions", json=payload, headers=headers)
                except httpx.TransportError as e:
                    kind: FailureKind = "transient"
                    last = f"network error: {type(e).__name__}"
                    body: Any = None
                else:
                    text = resp.text
                    try:
                        body = resp.json()
                    except ValueError:
                        body = None
                    base = body.get("base_resp") if isinstance(body, dict) else None
                    base_code = base.get("status_code") if isinstance(base, dict) else 0
                    if resp.status_code < 400 and not base_code:
                        return self._parse(body, attempt)
                    kind = classify_failure(resp.status_code, body, text)
                    last = f"HTTP {resp.status_code}: {_error_text(body, text)}"
                    retry_after = _retry_after(resp.headers.get("retry-after"))
            if kind == "quota":
                raise LLMQuotaExhausted(
                    f"{self.name}: quota or balance exhausted ({last}); not retrying",
                    provider=self.name,
                    reset_hint=_error_text(body, "") or None,
                )
            if kind == "auth":
                raise LLMAuthError(f"{self.name}: authentication failed ({last})")
            if kind == "fatal":
                raise LLMError(f"{self.name}: request rejected ({last})")
            if attempt < MAX_RETRIES:
                self._sleep(retry_after if retry_after is not None else _backoff(attempt))
        raise LLMRateLimited(f"{self.name}: still failing after {MAX_RETRIES} retries ({last})")

    def _parse(self, body: Any, attempt: int) -> ChatResult:
        try:
            choice = body["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{self.name}: unexpected response shape: {str(body)[:300]}") from e
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if not reasoning and isinstance(msg.get("reasoning_details"), list):
            reasoning = "\n".join(str(d.get("text", "")) for d in msg["reasoning_details"])
        in_t, out_t, cached = usage_of(body.get("usage") or {})
        finish = choice.get("finish_reason")
        return ChatResult(
            content=content,
            reasoning=reasoning,
            finish_reason=finish,
            in_tokens=in_t,
            out_tokens=out_t,
            cached_tokens=cached,
            retries=attempt,
            response={
                "id": body.get("id"),
                "model": body.get("model"),
                "finish_reason": finish,
                "content": content,
                "reasoning_content": reasoning,
                "usage": body.get("usage"),
            },
        )


def _error_text(body: Any, text: str) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        base = body.get("base_resp")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:300]
        if isinstance(base, dict) and base.get("status_msg"):
            return f"{base.get('status_code')} {base['status_msg']}"[:300]
    return text[:300]


def _retry_after(value: str | None) -> float | None:
    try:
        return min(float(value), BACKOFF_CAP_S) if value else None
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    delay: float = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2**attempt)
    return delay * (0.75 + random.random() / 2)


_SCHEMA_PROMPT = (
    "Reply with a single JSON value that conforms to the JSON Schema below. Output only "
    "the JSON: no commentary, no markdown fences.\n\nJSON Schema:\n"
)


def _with_schema(messages: list[Message], schema_json: str) -> list[Message]:
    """The schema instruction goes into the system message (created when absent). Keeping
    it ahead of the variable part helps providers' prefix caching."""
    instruction = _SCHEMA_PROMPT + schema_json
    if messages and messages[0].role == "system":
        first = Message("system", f"{messages[0].content}\n\n{instruction}", messages[0].images)
        return [first, *messages[1:]]
    return [Message("system", instruction), *messages]


def _repair_message(err: Exception) -> Message:
    if isinstance(err, ValidationError):
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in err.errors()[:10]
        )
    else:
        problems = str(err)
    return Message(
        "user",
        f"That reply was not acceptable: {problems}. "
        "Reply again with only the corrected JSON, conforming to the schema.",
    )


class OpenAICompatLLM:
    """Implements the `LLM` port. Each task's provider and model come from `cfg.tasks`."""

    def __init__(
        self,
        cfg: AppConfig,
        recorder: Recorder | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self._recorder = recorder
        self._transport = transport
        self._sleep = sleep
        self._clients: dict[str, ProviderClient] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        for c in self._clients.values():
            c.close()

    def _client(self, provider: str) -> ProviderClient:
        with self._lock:
            if provider not in self._clients:
                pcfg = self.cfg.providers[provider]
                self._clients[provider] = ProviderClient(
                    provider,
                    pcfg,
                    lambda: self.cfg.api_key(provider),
                    limiter=threading.BoundedSemaphore(pcfg.max_concurrency),
                    transport=self._transport,
                    sleep=self._sleep,
                )
            return self._clients[provider]

    def generate(
        self,
        task: str,
        messages: list[Message],
        schema: type[M],
        *,
        prompt_version: str,
        job_id: str | None = None,
        max_tokens: int = 4096,
    ) -> M:
        tcfg = self.cfg.tasks.get(task)
        if tcfg is None:
            raise LLMError(f"task {task!r} is not configured (see `tasks:` in config.yaml)")
        pcfg = self.cfg.providers[tcfg.provider]
        client = self._client(tcfg.provider)

        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
        extra: dict[str, Any] = {}
        if pcfg.json_mode == "native":
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": json.loads(schema_json)},
            }
            base_messages = messages
        else:
            base_messages = _with_schema(messages, schema_json)

        convo = base_messages
        last_error = ""
        for _ in range(2):  # the original ask, then one repair
            payload = {
                "model": tcfg.model,
                "messages": _wire_messages(convo),
                "max_tokens": max_tokens,
                **pcfg.extra_body,
                **extra,
            }
            started = time.monotonic()
            try:
                result = client.chat(payload)
            except LLMError as e:
                self._record(
                    task, tcfg.provider, tcfg.model, prompt_version, job_id, payload, started,
                    status="error", error=f"{type(e).__name__}: {e}",
                )  # fmt: skip
                raise
            try:
                value = schema.model_validate(extract_json(result.content))
            except (NoJsonFound, ValidationError) as e:
                last_error = str(e) if isinstance(e, NoJsonFound) else _repair_message(e).content
                self._record(
                    task, tcfg.provider, tcfg.model, prompt_version, job_id, payload, started,
                    result=result, status="error", error=f"schema: {last_error}"[:500],
                )  # fmt: skip
                if result.finish_reason == "length":
                    raise LLMSchemaError(
                        f"{task}: reply was cut off at max_tokens={max_tokens}; raise it "
                        "or ask for less"
                    ) from e
                convo = [*base_messages, Message("assistant", result.content), _repair_message(e)]
                continue
            self._record(
                task, tcfg.provider, tcfg.model, prompt_version, job_id, payload, started,
                result=result, status="ok",
            )  # fmt: skip
            return value
        raise LLMSchemaError(f"{task}: reply did not match {schema.__name__} twice: {last_error}")

    def _record(
        self,
        task: str,
        provider: str,
        model: str,
        prompt_version: str,
        job_id: str | None,
        payload: dict[str, Any],
        started: float,
        *,
        status: Literal["ok", "error"],
        result: ChatResult | None = None,
        error: str | None = None,
    ) -> None:
        if self._recorder is None:
            return
        self._recorder(
            LlmCallRecord(
                task=task,
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                status=status,
                job_id=job_id,
                error=error,
                retries=result.retries if result else 0,
                in_tokens=result.in_tokens if result else 0,
                out_tokens=result.out_tokens if result else 0,
                cached_tokens=result.cached_tokens if result else 0,
                latency_ms=round((time.monotonic() - started) * 1000),
                request=_redact_images(payload),
                response=result.response if result else {},
            )
        )
