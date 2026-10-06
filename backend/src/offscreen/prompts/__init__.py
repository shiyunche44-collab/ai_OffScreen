"""Prompt templates (`*.j2`). Each file starts with `{# version: N #}`; bump N whenever the
wording changes so the stage that uses it is invalidated (ARCHITECTURE §6.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_DIR = Path(__file__).parent
_VERSION = re.compile(r"\A\{#\s*version:\s*(\d+)\s*#\}")

_env = Environment(
    loader=FileSystemLoader(_DIR),
    undefined=StrictUndefined,  # a missing variable is a bug, not an empty string
    autoescape=False,  # prompts are plain text
    keep_trailing_newline=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass(frozen=True)
class Prompt:
    text: str
    version: str
    """`<template name>@<N>`; what `LLM.generate(prompt_version=...)` records."""


@cache
def template_version(name: str) -> str:
    source = (_DIR / f"{name}.j2").read_text(encoding="utf-8")
    m = _VERSION.match(source)
    if m is None:
        raise ValueError(f"prompt template {name}.j2 has no '{{# version: N #}}' header")
    return f"{name}@{m.group(1)}"


def render(name: str, **variables: Any) -> Prompt:
    return Prompt(
        _env.get_template(f"{name}.j2").render(**variables).strip(), template_version(name)
    )


def names() -> list[str]:
    """Every template, by name (the file name without `.j2`)."""
    return sorted(p.stem for p in _DIR.glob("*.j2"))


def variables(name: str) -> list[str]:
    """The variables a template reads (what `render` must be given)."""
    from jinja2 import meta

    ast = _env.parse((_DIR / f"{name}.j2").read_text(encoding="utf-8"))
    return sorted(meta.find_undeclared_variables(ast))


def preview(name: str, **values: Any) -> Prompt:
    """Like `render`, but a variable not given shows up as `{{ name }}` instead of failing, so a
    template's wording and layout can be read without data. For looking, never for sending."""
    from jinja2 import DebugUndefined

    env = _env.overlay(undefined=DebugUndefined)
    text = env.get_template(f"{name}.j2").render(**values).strip()
    return Prompt(text, template_version(name))
