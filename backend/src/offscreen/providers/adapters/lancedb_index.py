"""The shot index on LanceDB (embedded, no server). One table per index directory; vectors are
searched by cosine similarity with the filters applied first. A few thousand shots need no ANN
index: the scan is exact and instant."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa

from offscreen.providers.ports import IndexedShot, IndexHit, ShotFilter

TABLE = "shots"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def where_clause(f: ShotFilter) -> str | None:
    """The filter as a SQL condition (string values quoted, numbers formatted by us)."""
    parts: list[str] = []
    if f.scene_id is not None:
        parts.append(f"scene_id = {_quote(f.scene_id)}")
    if f.start_ms is not None:
        parts.append(f"start_ms >= {int(f.start_ms)}")
    if f.end_ms is not None:
        parts.append(f"end_ms <= {int(f.end_ms)}")
    if f.min_sharpness is not None:
        parts.append(f"sharpness >= {float(f.min_sharpness)!r}")
    if f.min_brightness is not None:
        parts.append(f"brightness >= {float(f.min_brightness)!r}")
    if f.exclude_credits:
        parts.append("is_credits = false")
    return " AND ".join(parts) or None


class LanceShotIndex:
    def build(self, path: Path, shots: Sequence[IndexedShot]) -> None:
        import lancedb

        if path.exists():
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns: dict[str, pa.Array] = {
            "shot_id": pa.array([s.shot_id for s in shots], pa.string()),
            "scene_id": pa.array([s.scene_id for s in shots], pa.string()),
            "start_ms": pa.array([s.start_ms for s in shots], pa.int64()),
            "end_ms": pa.array([s.end_ms for s in shots], pa.int64()),
            "sharpness": pa.array([s.sharpness for s in shots], pa.float32()),
            "brightness": pa.array([s.brightness for s in shots], pa.float32()),
            "is_credits": pa.array([s.is_credits for s in shots], pa.bool_()),
        }
        for name in ("image", "text"):
            vectors = [getattr(s, name) for s in shots]
            if shots and all(v is not None for v in vectors):
                dim = len(vectors[0])
                if any(len(v) != dim for v in vectors):
                    raise ValueError(f"the {name} vectors have different lengths")
                flat = pa.array([float(x) for v in vectors for x in v], pa.float32())
                columns[name] = pa.FixedSizeListArray.from_arrays(flat, dim)
        db = lancedb.connect(str(path))
        db.create_table(TABLE, pa.table(columns), mode="overwrite")

    def search(
        self,
        path: Path,
        column: Literal["image", "text"],
        vector: Sequence[float],
        *,
        limit: int,
        where: ShotFilter,
    ) -> list[IndexHit]:
        import lancedb

        table: Any = lancedb.connect(str(path)).open_table(TABLE)
        if column not in table.schema.names:
            raise LookupError(f"the index has no {column} vectors")
        query = table.search([float(x) for x in vector], vector_column_name=column).metric("cosine")
        clause = where_clause(where)
        if clause is not None:
            query = query.where(clause, prefilter=True)
        rows = query.limit(max(1, limit)).to_list()
        return [IndexHit(r["shot_id"], 1.0 - float(r["_distance"])) for r in rows]
