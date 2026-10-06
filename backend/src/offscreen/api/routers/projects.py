from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from offscreen.api.deps import Services
from offscreen.api.errors import ERROR_RESPONSES
from offscreen.domain.common import AssetId
from offscreen.domain.document import DocumentDiff, DocumentVersion
from offscreen.domain.job import Job
from offscreen.domain.project import Project, ProjectOptions
from offscreen.domain.script import OutlineBeat, Script, ScriptContent
from offscreen.services.library import ProjectDetail
from offscreen.services.outline import OutlineView

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


class SaveScript(ScriptContent):
    base_version: int | None = Field(
        description="The version this edit started from; null when the project has no script "
        "yet. 409 if it is no longer the current one."
    )


class RestoreScript(BaseModel):
    version: int = Field(ge=1, description="The old version to bring back as a new one.")
    base_version: int | None = Field(description="The current version, as for a save.")


@router.get("/{project_id}/script")
def get_script(project_id: str, services: Services, version: int | None = None) -> Script:
    """A version of the commentary text (default: the current one); 404 until there is one.
    Until the writing step stores its output in the document history, the generated draft is
    served as the current version."""
    if version is not None or services.documents.script_head(project_id) is not None:
        return services.documents.script(project_id, version)
    return services.library.script(project_id)


@router.put("/{project_id}/script")
def save_script(project_id: str, body: SaveScript, services: Services) -> Script:
    """Save an edit as a new version. `base_version` must be the current one (409 otherwise)."""
    content = ScriptContent.model_validate(body.model_dump(exclude={"base_version"}))
    return services.documents.save_script(project_id, content, body.base_version)


@router.get("/{project_id}/script/versions")
def script_versions(project_id: str, services: Services) -> list[DocumentVersion]:
    """The history, newest first."""
    return services.documents.script_versions(project_id)


@router.get("/{project_id}/script/diff")
def script_diff(project_id: str, a: int, b: int, services: Services) -> DocumentDiff:
    """Segment-by-segment difference between versions `a` and `b`."""
    return services.documents.diff_script(project_id, a, b)


@router.post("/{project_id}/script:restore")
def restore_script(project_id: str, body: RestoreScript, services: Services) -> Script:
    """Bring an old version back as a new one (the history only grows)."""
    return services.documents.restore_script(project_id, body.version, body.base_version)


class SaveOutline(BaseModel):
    beats: list[OutlineBeat] = Field(min_length=1)


@router.post("/{project_id}/outline:generate", status_code=202)
def generate_outline(project_id: str, services: Services) -> Job:
    """Propose the script's beats, scenes and timing (after the analysis is done)."""
    return services.outline.generate(project_id)


@router.get("/{project_id}/outline")
def get_outline(project_id: str, services: Services) -> OutlineView:
    """The outline: the person's edit if there is one, else the generated one; 404 until
    generated."""
    return services.outline.get(project_id)


@router.put("/{project_id}/outline")
def save_outline(project_id: str, body: SaveOutline, services: Services) -> OutlineView:
    """Edit the outline before the text is written. Kept apart from the generated one."""
    return services.outline.save(project_id, body.beats)


@router.delete("/{project_id}/outline")
def reset_outline(project_id: str, services: Services) -> OutlineView:
    """Drop the edit and return the generated outline."""
    return services.outline.reset(project_id)


@router.post("/{project_id}/script:generate", status_code=202)
def generate_script(project_id: str, services: Services) -> Job:
    return services.library.generate_script(project_id)


@router.post("/{project_id}/plan:build", status_code=202)
def build_plan(project_id: str, services: Services) -> Job:
    return services.library.build_plan(project_id)


@router.post("/{project_id}/render", status_code=202)
def render(project_id: str, services: Services) -> Job:
    return services.library.render(project_id)
