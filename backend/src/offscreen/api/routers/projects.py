from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from offscreen.api.deps import Services
from offscreen.api.errors import ERROR_RESPONSES
from offscreen.domain.common import AssetId
from offscreen.domain.job import Job
from offscreen.domain.project import Project, ProjectOptions
from offscreen.services.library import ProjectDetail

router = APIRouter(prefix="/projects", tags=["projects"], responses=ERROR_RESPONSES)


class CreateProject(BaseModel):
    asset_id: AssetId
    name: str | None = Field(default=None, description="Defaults to the asset's title.")
    options: ProjectOptions = Field(default_factory=ProjectOptions)


@router.post("", status_code=201)
def create_project(body: CreateProject, services: Services) -> Project:
    return services.library.create_project(body.asset_id, body.name, body.options)


@router.get("")
def list_projects(services: Services) -> list[Project]:
    return services.library.list_projects()


@router.get("/{project_id}")
def get_project(project_id: str, services: Services) -> ProjectDetail:
    """The project, which stages are built for its options, and the finished video if any."""
    return services.library.project(project_id)


@router.post("/{project_id}/script:generate", status_code=202)
def generate_script(project_id: str, services: Services) -> Job:
    return services.library.generate_script(project_id)


@router.post("/{project_id}/plan:build", status_code=202)
def build_plan(project_id: str, services: Services) -> Job:
    return services.library.build_plan(project_id)


@router.post("/{project_id}/render", status_code=202)
def render(project_id: str, services: Services) -> Job:
    return services.library.render(project_id)
