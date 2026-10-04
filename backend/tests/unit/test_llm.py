from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from offscreen.config import AppConfig
from offscreen.domain.llm import LlmCallRecord
from offscreen.providers.adapters.fake import FakeLLM
from offscreen.providers.adapters.openai_compat import (
    OpenAICompatLLM,
    classify_failure,
    usage_of,
)
from offscreen.providers.ports import (
    LLMAuthError,
    LLMError,
    LLMQuotaExhausted,
    LLMRateLimited,
    LLMSchemaError,
    Message,
)
from offscreen.services.llm import build_llm
from offscreen.store.db import Database
from offscreen.store.repos import LlmCallRepo

KEY = "sk-test-SECRET-123"


class Answer(BaseModel):
    answer: int
    why: str = ""


def make_cfg(tmp_path: Path, json_mode: str = "prompt", **provider: Any) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data_dir": str(tmp_path),
            "providers": {
                "p": {
                    "kind": "openai_compat",
                    "base_url": "https://llm.example/v1",
                    "api_key_env": "TEST_LLM_KEY",
                    "json_mode": json_mode,
                    "extra_body": {"reasoning_split": True},
                    **provider,
                }
            },
            "tasks": {"t": {"provider": "p", "model": "m-1"}},
            "asr": {"provider": "faster_whisper"},
            "tts": {"provider": "edge_tts"},
        }
    )


def completion(content: str, **extra: Any) -> dict[str, Any]:
    msg = {"role": "assistant", "content": content, **extra.pop("message", {})}
    return {
        "id": "cmpl-1",
        "model": "m-1",
        "choices": [{"index": 0, "message": msg, "finish_reason": extra.pop("finish", "stop")}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
        **extra,
    }


class Server:
    """A scripted endpoint: each item is a dict (200 JSON), an (status, body, headers) tuple,
    or an exception to raise from the transport."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            return httpx.Response(200, json=item)
        status, body, *headers = item
        content = body if isinstance(body, str) else json.dumps(body)
        return httpx.Response(status, content=content, headers=headers[0] if headers else {})

    def body(self, i: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[i].content)  # type: ignore[no-any-return]


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", KEY)


def build(
    tmp_path: Path, server: Server, **kw: Any
) -> tuple[OpenAICompatLLM, list[LlmCallRecord], list[float]]:
    records: list[LlmCallRecord] = []
    sleeps: list[float] = []
    llm = OpenAICompatLLM(
        make_cfg(tmp_path, **kw),
        records.append,
        transport=httpx.MockTransport(server),
        sleep=sleeps.append,
    )
    return llm, records, sleeps


def ask(llm: OpenAICompatLLM, **kw: Any) -> Answer:
    return llm.generate("t", [Message("user", "2+2?")], Answer, prompt_version="test@1", **kw)


# ---- structured output -------------------------------------------------------------------
def test_prompt_mode_puts_schema_in_system_message_and_parses_reply(tmp_path: Path) -> None:
    server = Server(completion('Sure:\n```json\n{"answer": 4}\n```'))
    llm, _, _ = build(tmp_path, server)
    assert ask(llm) == Answer(answer=4)

    req = server.body()
    assert req["model"] == "m-1" and req["reasoning_split"] is True  # extra_body merged
    assert req["messages"][0]["role"] == "system"
    assert (
        '"answer"' in req["messages"][0]["content"]
        and "JSON Schema" in req["messages"][0]["content"]
    )
    assert req["messages"][1] == {"role": "user", "content": "2+2?"}
    assert "response_format" not in req
    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert str(server.requests[0].url) == "https://llm.example/v1/chat/completions"


def test_existing_system_message_is_extended_not_duplicated(tmp_path: Path) -> None:
    server = Server(completion('{"answer": 1}'))
    llm, _, _ = build(tmp_path, server)
    llm.generate(
        "t", [Message("system", "Be brief."), Message("user", "hi")], Answer, prompt_version="v"
    )
    msgs = server.body()["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"].startswith("Be brief.\n\n")


def test_native_mode_uses_response_format(tmp_path: Path) -> None:
    server = Server(completion('{"answer": 4}'))
    llm, _, _ = build(tmp_path, server, json_mode="native")
    ask(llm)
    req = server.body()
    assert req["response_format"]["json_schema"]["name"] == "Answer"
    assert req["messages"] == [{"role": "user", "content": "2+2?"}]


def test_invalid_reply_gets_one_repair_round_with_the_errors(tmp_path: Path) -> None:
    server = Server(completion('{"answer": "four"}'), completion('{"answer": 4}'))
    llm, records, _ = build(tmp_path, server)
    assert ask(llm).answer == 4

    second = server.body(1)["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[2]["content"] == '{"answer": "four"}'
    assert "answer" in second[3]["content"] and "integer" in second[3]["content"].lower()
    assert [r.status for r in records] == ["error", "ok"]
    assert records[0].error and records[0].error.startswith("schema:")


def test_two_bad_replies_fail_with_schema_error(tmp_path: Path) -> None:
    server = Server(completion("not json"), completion('{"nope": 1}'))
    llm, records, _ = build(tmp_path, server)
    with pytest.raises(LLMSchemaError):
        ask(llm)
    assert len(server.requests) == 2 and [r.status for r in records] == ["error", "error"]


def test_truncated_reply_is_not_retried(tmp_path: Path) -> None:
    server = Server(completion('{"answer": 4, "why": "be', finish="length"))
    llm, _, _ = build(tmp_path, server)
    with pytest.raises(LLMSchemaError, match="max_tokens"):
        ask(llm, max_tokens=50)
    assert len(server.requests) == 1 and server.body()["max_tokens"] == 50


def test_unconfigured_task(tmp_path: Path) -> None:
    llm, _, _ = build(tmp_path, Server())
    with pytest.raises(LLMError, match="not configured"):
        llm.generate("nope", [Message("user", "x")], Answer, prompt_version="v")


def test_images_become_content_parts_and_are_redacted_in_the_record(tmp_path: Path) -> None:
    server = Server(completion('{"answer": 1}'))
    llm, records, _ = build(tmp_path, server)
    url = "data:image/jpeg;base64," + "A" * 5000
    llm.generate("t", [Message("user", "look", images=(url,))], Answer, prompt_version="v")

    sent = server.body()["messages"][1]["content"]
    assert sent[1] == {"type": "image_url", "image_url": {"url": url}}
    logged = json.dumps(records[0].request)
    assert "AAAA" not in logged and "<image" in logged


# ---- accounting --------------------------------------------------------------------------
def test_record_has_tokens_reasoning_and_no_secrets(tmp_path: Path) -> None:
    body = completion('{"answer": 4}', message={"reasoning_content": "2+2 is 4, easy"})
    llm, records, _ = build(tmp_path, Server(body))
    ask(llm, job_id="job_X")

    (rec,) = records
    assert (rec.task, rec.provider, rec.model, rec.prompt_version, rec.job_id) == (
        "t", "p", "m-1", "test@1", "job_X",
    )  # fmt: skip
    assert (rec.in_tokens, rec.out_tokens, rec.cached_tokens, rec.status) == (120, 30, 100, "ok")
    assert rec.response["reasoning_content"] == "2+2 is 4, easy"
    assert rec.latency_ms >= 0
    assert KEY not in rec.model_dump_json()


def test_calls_are_stored_in_llm_calls_with_body_files(tmp_path: Path) -> None:
    db = Database(tmp_path / "db" / "o.db")
    cfg = make_cfg(tmp_path)
    llm = build_llm(cfg, db)
    llm._transport = httpx.MockTransport(Server(completion('{"answer": 4}'), (402, "{}")))
    ask(llm, job_id="job_1")
    with pytest.raises(LLMQuotaExhausted):
        ask(llm)

    repo = LlmCallRepo(db, tmp_path)
    ok, err = repo.list()
    assert (ok.status, ok.job_id, ok.in_tokens, ok.cached_tokens) == ("ok", "job_1", 120, 100)
    assert err.status == "error" and "LLMQuotaExhausted" in (err.error or "")
    assert json.loads((tmp_path / (ok.req_path or "")).read_text())["model"] == "m-1"
    assert json.loads((tmp_path / (ok.resp_path or "")).read_text())["content"] == '{"answer": 4}'
    assert repo.list(job_id="job_1") == [ok]
    assert repo.totals() == {"calls": 2, "in_tokens": 120, "out_tokens": 30, "cached_tokens": 100}
    for f in (tmp_path / "llm_calls").iterdir():
        assert KEY not in f.read_text()
    db.close()


# ---- failures and retries ----------------------------------------------------------------
def test_429_is_retried_with_exponential_backoff_then_succeeds(tmp_path: Path) -> None:
    server = Server(
        (429, {"error": {"message": "Rate limit reached"}}),
        (503, "overloaded"),
        completion('{"answer": 4}'),
    )
    llm, records, sleeps = build(tmp_path, server)
    assert ask(llm).answer == 4
    assert len(server.requests) == 3 and records[0].retries == 2
    assert len(sleeps) == 2 and 1.4 < sleeps[0] < 2.6 and 2.9 < sleeps[1] < 5.1


def test_retry_after_header_is_honored(tmp_path: Path) -> None:
    server = Server((429, "slow down", {"retry-after": "7"}), completion('{"answer": 1}'))
    llm, _, sleeps = build(tmp_path, server)
    ask(llm)
    assert sleeps == [7.0]


def test_network_errors_are_retried(tmp_path: Path) -> None:
    server = Server(httpx.ConnectError("boom"), completion('{"answer": 1}'))
    llm, _, sleeps = build(tmp_path, server)
    assert ask(llm).answer == 1 and len(sleeps) == 1


def test_gives_up_after_three_retries(tmp_path: Path) -> None:
    server = Server(*[(429, "limit")] * 4)
    llm, records, sleeps = build(tmp_path, server)
    with pytest.raises(LLMRateLimited):
        ask(llm)
    assert len(server.requests) == 4 and len(sleeps) == 3
    assert records[0].status == "error" and "LLMRateLimited" in (records[0].error or "")


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (402, {"error": {"message": "Insufficient Balance"}}),
        (429, {"error": {"message": "You exceeded your current quota"}}),
        (200, {"base_resp": {"status_code": 2056, "status_msg": "usage limit exceeded"}}),
        (200, {"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}}),
    ],
)
def test_quota_exhaustion_fails_fast_without_retrying(
    tmp_path: Path, status: int, body: dict[str, Any]
) -> None:
    server = Server((status, body))
    llm, _, sleeps = build(tmp_path, server)
    with pytest.raises(LLMQuotaExhausted) as ei:
        ask(llm)
    assert len(server.requests) == 1 and sleeps == []
    assert ei.value.provider == "p"


def test_auth_errors_fail_fast(tmp_path: Path) -> None:
    llm, _, sleeps = build(tmp_path, Server((401, {"error": {"message": "bad key"}})))
    with pytest.raises(LLMAuthError):
        ask(llm)
    assert sleeps == []


def test_missing_api_key_is_a_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from offscreen.config import ConfigError

    monkeypatch.delenv("TEST_LLM_KEY")
    llm, _, _ = build(tmp_path, Server())
    with pytest.raises(ConfigError, match="TEST_LLM_KEY"):
        ask(llm)


def test_other_4xx_is_not_retried(tmp_path: Path) -> None:
    server = Server((400, {"error": {"message": "bad request"}}))
    llm, _, sleeps = build(tmp_path, server)
    with pytest.raises(LLMError, match="bad request"):
        ask(llm)
    assert sleeps == []


def test_malformed_success_body(tmp_path: Path) -> None:
    llm, _, _ = build(tmp_path, Server({"choices": []}))
    with pytest.raises(LLMError, match="unexpected response"):
        ask(llm)


@pytest.mark.parametrize(
    ("status", "body", "text", "kind"),
    [
        (429, None, "Too Many Requests", "transient"),
        (429, None, "额度已用完", "quota"),
        (500, None, "", "transient"),
        (401, None, "", "auth"),
        (200, {"base_resp": {"status_code": 1039}}, "", "transient"),
        (200, {"base_resp": {"status_code": 1004}}, "", "auth"),
        (200, {"base_resp": {"status_code": 2013}}, "", "fatal"),
        (404, None, "", "fatal"),
    ],
)
def test_classify_failure(status: int, body: Any, text: str, kind: str) -> None:
    assert classify_failure(status, body, text) == kind


def test_usage_variants() -> None:
    assert usage_of({"prompt_tokens": 5, "completion_tokens": 2, "prompt_cache_hit_tokens": 3}) == (
        5, 2, 3,
    )  # fmt: skip
    assert usage_of({}) == (0, 0, 0)


def test_concurrency_limit_is_per_provider(tmp_path: Path) -> None:
    import threading
    import time

    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def handler(_req: httpx.Request) -> httpx.Response:
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        with lock:
            active["now"] -= 1
        return httpx.Response(200, json=completion('{"answer": 1}'))

    llm = OpenAICompatLLM(
        make_cfg(tmp_path, max_concurrency=2), transport=httpx.MockTransport(handler)
    )
    threads = [threading.Thread(target=lambda: ask(llm)) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert active["max"] == 2


# ---- fake adapter ------------------------------------------------------------------------
def test_fake_llm_scripts_replies_validates_and_records() -> None:
    records: list[LlmCallRecord] = []
    fake = FakeLLM(
        {
            "t": [Answer(answer=1), {"answer": 2}, '```json\n{"answer": 3}\n```', LLMError("down")],
            "f": lambda task, msgs, schema: {"answer": len(msgs)},
        },
        recorder=records.append,
    )
    got = [
        fake.generate("t", [Message("user", "x")], Answer, prompt_version="v").answer
        for _ in range(3)
    ]
    assert got == [1, 2, 3]
    with pytest.raises(LLMError, match="down"):
        fake.generate("t", [Message("user", "x")], Answer, prompt_version="v")
    with pytest.raises(LLMError, match="no scripted reply"):
        fake.generate("t", [Message("user", "x")], Answer, prompt_version="v")
    two = [Message("user", "a"), Message("user", "b")]
    assert fake.generate("f", two, Answer, prompt_version="v").answer == 2
    assert [c[0] for c in fake.calls][-1] == "f" and len(records) == 4
    assert records[0].provider == "fake"


def test_fake_llm_rejects_replies_that_break_the_schema() -> None:
    from pydantic import ValidationError

    fake = FakeLLM({"t": [{"answer": "x"}]})
    with pytest.raises(ValidationError):
        fake.generate("t", [Message("user", "x")], Answer, prompt_version="v")
