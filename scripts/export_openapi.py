"""Write the API's OpenAPI document to docs/openapi.json. Run via `make openapi`.
The frontend types are generated from this file (ARCHITECTURE R7)."""

import json
from pathlib import Path

from offscreen.api.app import create_app

out = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"
out.write_text(
    json.dumps(create_app().openapi(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(f"wrote {out}")
