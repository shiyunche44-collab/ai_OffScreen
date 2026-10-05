from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from offscreen.api.deps import Services
from offscreen.api.errors import ERROR_RESPONSES
from offscreen.domain.asset import MediaAsset
from offscreen.domain.job import Job
from offscreen.services.library import AssetDetail

router = APIRouter(prefix="/assets", tags=["assets"], responses=ERROR_RESPONSES)


class ImportAsset(BaseModel):
    path: str = Field(min_length=1, description="A movie file under one of the media roots.")
    title: str | None = None


@router.post("", status_code=201)
def import_asset(body: ImportAsset, services: Services) -> MediaAsset:
    """Register a movie. Importing the same file again returns the same asset."""
    return services.library.import_asset(body.path, body.title)


@router.get("")
def list_assets(services: Services) -> list[MediaAsset]:
    return services.library.list_assets()


@router.get("/{asset_id}")
def get_asset(asset_id: str, services: Services) -> AssetDetail:
    """The asset and which analysis stages are already built."""
    return services.library.asset(asset_id)


@router.post("/{asset_id}/analyze", status_code=202)
def analyze_asset(asset_id: str, services: Services) -> Job:
    """Queue the analysis. Returns the job (the already active one, if there is one)."""
    return services.library.analyze(asset_id)
