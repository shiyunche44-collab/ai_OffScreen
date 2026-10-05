from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from offscreen.api.deps import Services
from offscreen.api.errors import ERROR_RESPONSES
from offscreen.domain.asset import MediaAsset
from offscreen.domain.index import Scenes, Story, Transcript
from offscreen.domain.job import Job
from offscreen.services.characters import CharactersView
from offscreen.services.index import ShotsView
from offscreen.services.library import AssetDetail, MediaListing
from offscreen.services.report import AnalysisReport

router = APIRouter(prefix="/assets", tags=["assets"], responses=ERROR_RESPONSES)


class ImportAsset(BaseModel):
    path: str = Field(min_length=1, description="A movie file under one of the media roots.")
    title: str | None = None


@router.post("", status_code=201)
def import_asset(body: ImportAsset, services: Services) -> MediaAsset:
    """Register a movie. Importing the same file again returns the same asset."""
    return services.library.import_asset(body.path, body.title)


@router.get("")
def list_assets(services: Services) -> list[AssetDetail]:
    """Every asset with which analysis stages are already built."""
    return services.library.list_assets()


@router.get("/browse")
def browse_media(services: Services, path: str | None = None) -> MediaListing:
    """Folders and video files under the media roots (no `path`: the roots), for picking a movie."""
    return services.library.browse(path)


@router.get("/{asset_id}")
def get_asset(asset_id: str, services: Services) -> AssetDetail:
    """The asset and which analysis stages are already built."""
    return services.library.asset(asset_id)


@router.post("/{asset_id}/analyze", status_code=202)
def analyze_asset(asset_id: str, services: Services) -> Job:
    """Queue the analysis. Returns the job (the already active one, if there is one)."""
    return services.library.analyze(asset_id)


@router.get("/{asset_id}/report")
def analysis_report(asset_id: str, services: Services) -> AnalysisReport:
    """What the analysis cost: time and model usage per stage, and what it found."""
    return services.report.analysis(asset_id)


@router.get("/{asset_id}/index/transcript")
def index_transcript(asset_id: str, services: Services) -> Transcript:
    """The dialogue lines with their times (404 until the transcript stage is built)."""
    return services.index.transcript(asset_id)


@router.get("/{asset_id}/index/shots")
def index_shots(asset_id: str, services: Services) -> ShotsView:
    """Shots with keyframes, quality, description and sprite position, plus the proxy video."""
    return services.index.shots(asset_id)


@router.get("/{asset_id}/index/scenes")
def index_scenes(asset_id: str, services: Services) -> Scenes:
    """The scenes (404 until the scenes stage is built)."""
    return services.index.scenes(asset_id)


@router.get("/{asset_id}/index/story")
def index_story(asset_id: str, services: Services) -> Story:
    """The story: logline, acts, turning points, ending (404 until the story stage is built)."""
    return services.index.story(asset_id)


@router.get("/{asset_id}/index/characters")
def index_characters(asset_id: str, services: Services) -> CharactersView:
    """The characters with human edits applied (404 until the characters stage is built)."""
    return services.characters.view(asset_id)
