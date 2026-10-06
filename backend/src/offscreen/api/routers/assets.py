from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from offscreen.api.deps import Services
from offscreen.api.errors import ERROR_RESPONSES
from offscreen.domain.asset import MediaAsset
from offscreen.domain.index import Scenes, Story, Transcript
from offscreen.domain.job import Job
from offscreen.providers.ports import ShotFilter
from offscreen.services.annotations import CutEvaluation, CutsView
from offscreen.services.characters import CharacterEdit, CharactersView
from offscreen.services.index import ShotsView
from offscreen.services.library import AssetDetail, MediaListing
from offscreen.services.report import AnalysisReport
from offscreen.services.search import ShotSearchResult

router = APIRouter(prefix="/assets", tags=["assets"], responses=ERROR_RESPONSES)


class MarkedCuts(BaseModel):
    cuts: list[int] = Field(description="First frame of each new shot, from 0; at least 1.")


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


@router.post("/{asset_id}/characters:build", status_code=202)
def build_characters(asset_id: str, services: Services) -> Job:
    """Queue face detection, grouping into people and naming (with the analysis it needs).
    Returns the job (the already active one, if there is one)."""
    return services.library.identify_characters(asset_id)


@router.patch("/{asset_id}/characters/{character_id}")
def edit_character(
    asset_id: str, character_id: str, change: CharacterEdit, services: Services
) -> CharactersView:
    """Rename, ignore, merge or reset one character. Edits go to the revision layer; the AI
    output is untouched. Returns the characters as they now read."""
    return services.characters.edit(asset_id, character_id, change)


@router.get("/{asset_id}/shots/search")
def search_shots(
    asset_id: str,
    services: Services,
    q: str = Query(min_length=1, description="What to look for, in words (Chinese or English)."),
    limit: int = Query(default=10, ge=1, le=100),
    scene_id: str | None = None,
    start_ms: int | None = Query(default=None, ge=0, description="Shots starting at or after."),
    end_ms: int | None = Query(default=None, ge=0, description="Shots ending at or before."),
    min_sharpness: float | None = Query(default=None, ge=0, le=1),
    min_brightness: float | None = Query(default=None, ge=0, le=1),
    exclude_credits: bool = True,
) -> ShotSearchResult:
    """Shots matching a description, best first (404 until the embeddings stage is built)."""
    return services.search.search(
        asset_id,
        q,
        limit=limit,
        where=ShotFilter(
            scene_id=scene_id,
            start_ms=start_ms,
            end_ms=end_ms,
            min_sharpness=min_sharpness,
            min_brightness=min_brightness,
            exclude_credits=exclude_credits,
        ),
    )


@router.get("/{asset_id}/annotations/cuts")
def get_marked_cuts(asset_id: str, services: Services) -> CutsView:
    """The hard cuts a person marked in this movie (empty until saved), with its frame rate."""
    return services.annotations.cuts(asset_id)


@router.put("/{asset_id}/annotations/cuts")
def put_marked_cuts(asset_id: str, body: MarkedCuts, services: Services) -> CutsView:
    """Replace the marked cuts. Order and duplicates are tidied."""
    return services.annotations.save_cuts(asset_id, body.cuts)


@router.get("/{asset_id}/annotations/cuts/evaluation")
def evaluate_marked_cuts(
    asset_id: str, services: Services, tolerance: int = Query(default=2, ge=0, le=50)
) -> CutEvaluation:
    """Precision / recall / F1 of the detected shot boundaries against the marked cuts (404
    before either exists). `tolerance` is in frames."""
    return services.annotations.evaluate_cuts(asset_id, tolerance=tolerance)
