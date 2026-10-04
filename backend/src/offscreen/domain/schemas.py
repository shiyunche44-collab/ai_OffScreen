"""JSON Schema export for every document type (docs/schemas/*.schema.json)."""

from __future__ import annotations

import json

from offscreen.domain.registry import DOCUMENTS


def export_schemas() -> dict[str, str]:
    """name -> deterministic JSON Schema text."""
    return {
        name: json.dumps(cls.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
        for name, cls in DOCUMENTS.items()
    }
