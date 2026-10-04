"""Use case: show the effective configuration (secrets redacted)."""

from __future__ import annotations

from pathlib import Path

import yaml

from offscreen.config import load_config, redacted


def show_config(path: Path | None = None) -> str:
    return yaml.safe_dump(redacted(load_config(path)), allow_unicode=True, sort_keys=False)
