from __future__ import annotations

from fastapi import APIRouter

from offscreen.api.deps import Services
from offscreen.api.errors import ERROR_RESPONSES
from offscreen.domain.style import StylePreset

router = APIRouter(prefix="/styles", tags=["styles"], responses=ERROR_RESPONSES)


@router.get("")
def list_styles(services: Services) -> list[StylePreset]:
    """The writing style presets a project can choose from."""
    return services.library.styles()
