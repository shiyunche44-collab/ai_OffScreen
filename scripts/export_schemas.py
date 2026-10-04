"""Write JSON Schemas to docs/schemas/. Run via `make schemas`."""

from pathlib import Path

from offscreen.domain.schemas import export_schemas

out = Path(__file__).resolve().parents[1] / "docs" / "schemas"
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("*.schema.json"):
    old.unlink()
for name, text in export_schemas().items():
    (out / f"{name}.schema.json").write_text(text, encoding="utf-8")
print(f"wrote {len(list(out.glob('*.schema.json')))} schemas to {out}")
