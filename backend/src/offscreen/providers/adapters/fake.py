"""Scripted LLM for tests and offline runs: no network, deterministic, inspectable."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel

from offscreen.algo.jsonreply import extract_json
from offscreen.domain.llm import LlmCallRecord
from offscreen.providers.ports import LLMError, M, Message, Recorder

Reply = BaseModel | dict[str, Any] | str | Exception
"""A model instance or dict (validated against the requested schema), a string (parsed
like a real reply, so fences and prose are exercised), or an exception to raise."""
ReplyFn = Callable[[str, list[Message], type[BaseModel]], Reply]


class FakeLLM:
    """`replies[task]` is a list used in order, or a function of (task, messages, schema)."""

    def __init__(
        self,
        replies: Mapping[str, list[Reply] | ReplyFn],
        recorder: Recorder | None = None,
    ) -> None:
        self._queues: dict[str, deque[Reply]] = defaultdict(deque)
        self._fns: dict[str, ReplyFn] = {}
        for task, r in replies.items():
            if callable(r):
                self._fns[task] = r
            else:
                self._queues[task].extend(r)
        self._recorder = recorder
        self.calls: list[tuple[str, list[Message], str]] = []
        """(task, messages, prompt_version) for every generate() call."""

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
        self.calls.append((task, list(messages), prompt_version))
        if task in self._fns:
            reply = self._fns[task](task, messages, schema)
        elif self._queues[task]:
            reply = self._queues[task].popleft()
        else:
            raise LLMError(f"FakeLLM: no scripted reply left for task {task!r}")
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            value = schema.model_validate(extract_json(reply))
        elif isinstance(reply, BaseModel):
            value = schema.model_validate(reply.model_dump())
        else:
            value = schema.model_validate(reply)
        if self._recorder is not None:
            self._recorder(
                LlmCallRecord(
                    task=task,
                    provider="fake",
                    model="fake",
                    prompt_version=prompt_version,
                    status="ok",
                    job_id=job_id,
                    request={
                        "messages": [{"role": m.role, "content": m.content} for m in messages]
                    },
                    response={"content": value.model_dump_json()},
                )
            )
        return value
