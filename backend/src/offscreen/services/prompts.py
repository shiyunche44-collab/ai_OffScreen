"""Looking at the prompt templates (what the models are actually told)."""

from __future__ import annotations

import json
from typing import Any

from offscreen import prompts
from offscreen.services.errors import InvalidInput, NotFound


def list_prompts() -> list[tuple[str, str, list[str]]]:
    """`(name, version, variables)` of every template."""
    return [(n, prompts.template_version(n), prompts.variables(n)) for n in prompts.names()]


def render_prompt(name: str, vars_json: str | None = None) -> tuple[str, str]:
    """`(version, text)` of a template. Variables come from a JSON object; any not given appear
    as `{{ name }}` in the text."""
    if name not in prompts.names():
        raise NotFound(f"unknown prompt {name!r}; known: {', '.join(prompts.names())}")
    values: dict[str, Any] = {}
    if vars_json:
        try:
            parsed = json.loads(vars_json)
        except json.JSONDecodeError as e:
            raise InvalidInput(f"the variables are not valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise InvalidInput("the variables must be a JSON object")
        values = parsed
    shown = prompts.preview(name, **values)
    return shown.version, shown.text
