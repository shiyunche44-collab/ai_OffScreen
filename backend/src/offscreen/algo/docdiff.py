"""Segment-level diff of two versions of a script or plan.

Segments are matched by id (ids are stable across versions, so an edited text is one `changed`
segment, not a removal plus an addition). Works on plain dicts (`model_dump`) so it serves both
document kinds."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from offscreen.domain.document import DocKind, DocumentDiff, SegmentChange

# Server-managed fields: they differ between any two versions and say nothing about content.
_META = {"id", "version", "parent_version", "author", "schema_version", "segments"}


def diff_documents(
    kind: DocKind, a: int, b: int, doc_a: Mapping[str, Any], doc_b: Mapping[str, Any]
) -> DocumentDiff:
    segs_a = {s["id"]: s for s in doc_a["segments"]}
    segs_b = {s["id"]: s for s in doc_b["segments"]}
    order_a = [i for i in segs_a if i in segs_b]
    order_b = [i for i in segs_b if i in segs_a]

    changes = [_change(i, segs_a.get(i), segs_b[i]) for i in segs_b]
    changes += [SegmentChange(segment_id=i, status="removed") for i in segs_a if i not in segs_b]
    params = sorted(k for k in (set(doc_a) | set(doc_b)) - _META if doc_a.get(k) != doc_b.get(k))
    return DocumentDiff(
        kind=kind,
        a=a,
        b=b,
        reordered=order_a != order_b,
        changes=changes,
        params_changed=params,
    )


def _change(
    segment_id: str, old: Mapping[str, Any] | None, new: Mapping[str, Any]
) -> SegmentChange:
    if old is None:
        return SegmentChange(segment_id=segment_id, status="added")
    fields = sorted(k for k in (set(old) | set(new)) if old.get(k) != new.get(k))
    if not fields:
        return SegmentChange(segment_id=segment_id, status="unchanged")
    return SegmentChange(segment_id=segment_id, status="changed", fields=fields)
