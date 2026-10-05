"""A project: one asset plus the settings its commentary is made with."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from offscreen.domain.common import AssetId, ProjectId, Strict


class ProjectOptions(Strict):
    """How the commentary is written and voiced. Stages read these; changing them changes the
    creative stages' cache keys, never the analysis."""

    minutes: float = Field(default=3.0, gt=0, le=240)
    """Target length of the commentary."""
    voice: str | None = Field(default=None, min_length=1)
    """Voice id; None: the configured default."""
    style: str = Field(default="neutral", min_length=1)
    spoil_ending: bool = True


class Project(Strict):
    id: ProjectId
    asset_id: AssetId
    name: str = Field(min_length=1)
    options: ProjectOptions = Field(default_factory=ProjectOptions)
    created_at: datetime
