"""ingest: register a movie file as an asset (L0). ARCHITECTURE §7.1.

Entry point of the pipeline rather than a cached Stage: it is what *creates* the
`asset_id` every other stage is scoped by, and its result is a database row. The source
file is never copied; we keep its path and fingerprint, and relink by fingerprint if the
file is moved."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from offscreen.domain.asset import MediaAsset
from offscreen.domain.common import new_id
from offscreen.media.probe import ProbeError, ProbeResult, probe
from offscreen.store.files import fingerprint_file
from offscreen.store.repos import AssetRepo

SUBTITLE_EXTS = (".srt", ".ass", ".ssa")


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class IngestResult:
    asset: MediaAsset
    created: bool
    """False when the file was already registered (same fingerprint)."""
    relinked: bool = False
    """True when a known asset was found at a new path and its path was updated."""


def find_external_subtitles(video: Path) -> str | None:
    """A sibling `<stem>.srt|.ass|.ssa`, if any (first extension in that order wins)."""
    for ext in SUBTITLE_EXTS:
        cand = video.with_suffix(ext)
        if cand.is_file():
            return str(cand)
    return None


def ingest(
    path: Path,
    repo: AssetRepo,
    *,
    probe_fn: Callable[[Path], ProbeResult] = probe,
    title: str | None = None,
) -> IngestResult:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise IngestError(f"not a file: {source}")

    fingerprint = fingerprint_file(source)
    existing = repo.find_by_fingerprint(fingerprint)
    if existing is not None:
        if existing.source_path == str(source) or Path(existing.source_path).is_file():
            return IngestResult(existing, created=False)
        moved = existing.model_copy(
            update={
                "source_path": str(source),
                "subtitles_external": find_external_subtitles(source),
            }
        )
        repo.update(moved)
        return IngestResult(moved, created=False, relinked=True)

    try:
        info = probe_fn(source)
    except ProbeError as e:
        raise IngestError(str(e)) from e

    asset = MediaAsset(
        id=new_id("ast"),
        title=title or source.stem,
        source_path=str(source),
        fingerprint=fingerprint,
        duration_ms=info.duration_ms,
        video=info.video,
        audio=info.audio,
        subtitles_external=find_external_subtitles(source),
    )
    # add() is idempotent on fingerprint, so a concurrent import of the same file still
    # converges on one asset.
    stored = repo.add(asset)
    return IngestResult(stored, created=stored.id == asset.id)
