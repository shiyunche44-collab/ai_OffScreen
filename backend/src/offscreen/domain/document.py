"""Versioned documents (script, plan): which versions exist, not their content."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from offscreen.domain.common import Strict
from offscreen.domain.script import Author

DocKind = Literal["script", "plan"]


class DocumentVersion(Strict):
    """One entry of a document's history."""

    kind: DocKind
    version: int
    parent_version: int | None
    author: Author
    created_at: datetime


class SegmentChange(Strict):
    """How one segment differs between two versions of a document."""

    segment_id: str
    status: Literal["added", "removed", "changed", "unchanged"]
    fields: list[str] = []
    """For `changed`: the segment fields whose values differ (e.g. `text`, `scene_refs`)."""


class DocumentDiff(Strict):
    kind: DocKind
    a: int
    b: int
    reordered: bool
    """The segments both versions have appear in a different order."""
    changes: list[SegmentChange]
    """Segments of version `b` in their order, then those only `a` had."""
    params_changed: list[str] = []
    """Top-level fields other than `segments` that differ (script params, outline, …)."""
