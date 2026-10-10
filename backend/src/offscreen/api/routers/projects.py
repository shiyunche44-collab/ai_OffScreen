from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from offscreen.api.deps import Services
from offscreen.api.errors import ERROR_RESPONSES
from offscreen.domain.common import AssetId
from offscreen.domain.document import DocumentDiff, DocumentVersion
from offscreen.domain.job import Job
from offscreen.domain.plan import EditPlan
from offscreen.domain.plan_edit import PlanOp
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


class UpdateProject(BaseModel):
    name: str | None = Field(default=None, description="Leave out to keep the name.")
    options: ProjectOptions | None = Field(
        default=None, description="The full options to use from now on; leave out to keep them."
    )


@router.get("")
def list_projects(services: Services) -> list[Project]:
    return services.library.list_projects()


@router.patch("/{project_id}")
def update_project(project_id: str, body: UpdateProject, services: Services) -> Project:
    """Rename a project or change its options. What is already built stays; the creative
    steps are rebuilt with the new options the next time they run."""
    return services.library.update_project(project_id, body.name, body.options)


@router.get("/{project_id}")
def get_project(project_id: str, services: Services) -> ProjectDetail:
    """The project, which stages are built for its options, and the finished video if any."""
    return services.library.project(project_id)


class SaveScript(ScriptContent):
    base_version: int | None = Field(
        description="The version this edit started from; null when the project has no script "
        "yet. 409 if it is no longer the current one."
    )


class RewriteSegment(BaseModel):
    instruction: str = Field(
        min_length=1, max_length=200, description='How it should change, e.g. "more colloquial".'
    )
    base_version: int = Field(ge=1, description="The script version being edited (409 if stale).")


class RestoreScript(BaseModel):
    version: int = Field(ge=1, description="The old version to bring back as a new one.")
    base_version: int | None = Field(description="The current version, as for a save.")


@router.get("/{project_id}/script")
def get_script(project_id: str, services: Services, version: int | None = None) -> Script:
    """A version of the commentary text (default: the current one); 404 until there is one."""
    return services.documents.script(project_id, version)


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


@router.post("/{project_id}/script/segments/{segment_id}:rewrite", status_code=202)
def rewrite_segment(
    project_id: str, segment_id: str, body: RewriteSegment, services: Services
) -> Job:
    """Rewrite one narration segment as instructed. The job stores the result as the next
    script version (author ai); the other segments are untouched."""
    return services.library.rewrite_segment(
        project_id, segment_id, body.instruction, body.base_version
    )


@router.post("/{project_id}/script:restore")
def restore_script(project_id: str, body: RestoreScript, services: Services) -> Script:
    """Bring an old version back as a new one (the history only grows)."""
    return services.documents.restore_script(project_id, body.version, body.base_version)


class EditPlanBody(BaseModel):
    base_version: int = Field(ge=1, description="The plan version being edited (409 if stale).")
    ops: list[PlanOp] = Field(
        min_length=1,
        description="Applied in order, all or nothing; clips are addressed by position in "
        "their segment. Picked or trimmed footage is locked.",
    )


@router.get("/{project_id}/plan")
def get_plan(project_id: str, services: Services, version: int | None = None) -> EditPlan:
    """A version of the edit plan (default: the current one); 404 until it is built."""
    return services.documents.plan(project_id, version)


@router.get("/{project_id}/plan/versions")
def plan_versions(project_id: str, services: Services) -> list[DocumentVersion]:
    """The history, newest first."""
    return services.documents.plan_versions(project_id)


@router.get("/{project_id}/plan/diff")
def plan_diff(project_id: str, a: int, b: int, services: Services) -> DocumentDiff:
    """Segment-by-segment difference between versions `a` and `b`."""
    return services.documents.diff_plan(project_id, a, b)


@router.post("/{project_id}/plan:edit")
def edit_plan(project_id: str, body: EditPlanBody, services: Services) -> EditPlan:
    """Swap, trim, lock, add or remove footage; move, delete or insert segments; change a
    voice. Stored as the next version (author human). A new voice is spoken by the next build."""
    return services.plan_edits.edit(project_id, body.ops, body.base_version)


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
