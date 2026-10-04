"""Pull a JSON value out of a chatty model reply. Pure."""

from __future__ import annotations

import json
import re
from typing import Any

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


class NoJsonFound(ValueError):
    pass


def extract_json(text: str) -> Any:
    """The first JSON object or array in `text`.

    Handles inline `<think>` blocks (some models ignore reasoning separation), markdown
    fences, and prose before or after the value. A fenced block is preferred over loose
    text because models quote braces in their explanations."""
    text = _THINK.sub("", text)
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        for i, ch in enumerate(chunk):
            if ch in "{[":
                try:
                    value, _ = decoder.raw_decode(chunk[i:])
                except json.JSONDecodeError:
                    continue
                return value
    raise NoJsonFound("no JSON object or array found in the reply")
