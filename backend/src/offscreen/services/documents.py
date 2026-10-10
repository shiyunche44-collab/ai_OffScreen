"""Versioned documents: save with `base_version`, read any version, compare, restore.

Only the script has an editor so far (M4); the plan joins in M5 through the same store: the
build reads the current script and plan from it and stores what it makes as the next plan
version (`adopt_generated_plan`)."""

from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import ValidationError

from offscreen.algo.docdiff import diff_documents
from offscreen.algo.plan_build import same_content
from offscreen.config import AppConfig
from offscreen.domain.document import DocumentDiff, DocumentVersion
from offscreen.domain.plan import EditPlan
from offscreen.domain.script import Author, Script, ScriptContent
from offscreen.services.errors import Conflict, InvalidInput, NotFound
from offscreen.stages.creation.plan import PlanSettings, PreviousPlan
from offscreen.store.db import Database
from offscreen.store.documents import DocumentStore, StaleBase
from offscreen.store.repos import ProjectRepo


class DocumentService:
    def __init__(self, cfg: AppConfig, db: Database) -> None:
        self.store = DocumentStore(db, cfg.data_dir)
        self.projects = ProjectRepo(db)

    def script_head(self, project_id: str) -> int | None:
        self._project(project_id)
        return self.store.head(project_id, "script")

    def script(self, project_id: str, version: int | None = None) -> Script:
        """A version of the script (default: the current one)."""
        self._project(project_id)
        doc = self.store.read(project_id, "script", Script, version)
        if doc is None:
            raise NotFound(
                "the project has no script yet"
                if version is None
                else f"the script has no version {version}"
            )
        return doc

    def script_versions(self, project_id: str) -> list[DocumentVersion]:
        self._project(project_id)
        return self.store.versions(project_id, "script")

    def save_script(
        self,
        project_id: str,
        content: ScriptContent,
        base_version: int | None,
        *,
        author: Author = "human",
    ) -> Script:
        """Store `content` as the version after `base_version` (None: the first). Conflict when
        `base_version` is no longer the current one."""
        self._project(project_id)

        def build(doc_id: str, version: int, parent: int | None) -> Script:
            try:
                return Script(
                    id=doc_id,
                    project_id=project_id,
                    version=version,
                    parent_version=parent,
                    author=author,
                    **{k: getattr(content, k) for k in ScriptContent.model_fields},
                )
            except ValidationError as e:
                raise InvalidInput(f"the script is not valid: {e}") from e

        try:
            return self.store.append(project_id, "script", base_version, build, author=author)
        except StaleBase as e:
            raise Conflict(
                f"the script was updated (base_version {base_version}, "
                f"current {e.current}); reload it before saving"
            ) from e

    def restore_script(self, project_id: str, version: int, base_version: int | None) -> Script:
        """Bring back an old version as a new one (history only grows)."""
        return self.save_script(
            project_id, ScriptContent.of(self.script(project_id, version)), base_version
        )

    def diff_script(self, project_id: str, a: int, b: int) -> DocumentDiff:
        old, new = self.script(project_id, a), self.script(project_id, b)
        return diff_documents("script", a, b, _dump(old), _dump(new))

    def plan_head(self, project_id: str) -> int | None:
        self._project(project_id)
        return self.store.head(project_id, "plan")

    def plan(self, project_id: str, version: int | None = None) -> EditPlan:
        """A version of the edit plan (default: the current one)."""
        self._project(project_id)
        doc = self.store.read(project_id, "plan", EditPlan, version)
        if doc is None:
            raise NotFound(
                "the project has no plan yet"
                if version is None
                else f"the plan has no version {version}"
            )
        return doc

    def plan_versions(self, project_id: str) -> list[DocumentVersion]:
        self._project(project_id)
        return self.store.versions(project_id, "plan")

    def diff_plan(self, project_id: str, a: int, b: int) -> DocumentDiff:
        old, new = self.plan(project_id, a), self.plan(project_id, b)
        return diff_documents(
            "plan", a, b, old.model_dump(mode="json"), new.model_dump(mode="json")
        )

    def _project(self, project_id: str) -> None:
        if self.projects.get(project_id) is None:
            raise NotFound(f"project {project_id} not found")


def _dump(script: Script) -> dict[str, object]:
    return script.model_dump(mode="json")


def adopt_generated_script(
    store: DocumentStore, project_id: str, draft: Script, base_version: int | None
) -> Script:
    """Store an AI-written script as the project's next version. `base_version` is the head the
    request was made against: if a person saved something meanwhile, nothing is stored and the
    job fails (the draft stays cached, so pressing generate again is cheap). A draft identical
    to the current version is not stored twice."""
    head = store.read(project_id, "script", Script)
    if head is not None and ScriptContent.of(head) == ScriptContent.of(draft):
        return head

    def build(doc_id: str, version: int, parent: int | None) -> Script:
        return draft.model_copy(
            update={
                "id": doc_id,
                "project_id": project_id,
                "version": version,
                "parent_version": parent,
                "author": "ai",
            }
        )

    try:
        return store.append(project_id, "script", base_version, build, author="ai")
    except StaleBase as e:
        raise Conflict(
            f"the script changed while it was being generated (it was version {base_version}, "
            f"now {e.current}); generate again to build on the current one"
        ) from e


def plan_settings_for(store: DocumentStore, project_id: str) -> PlanSettings:
    """What the plan build starts from: the project's current script (a person may have edited
    it) and its current plan, if there is one (the build then redoes only what changed). The
    script the plan was built from tells which structural differences are the person's edits of
    the plan and which are changes of the script."""
    plan = store.read(project_id, "plan", EditPlan)
    built_from = store.read(project_id, "script", Script, plan.script_ref.version) if plan else None
    return PlanSettings(
        script=store.read(project_id, "script", Script),
        previous=PreviousPlan(plan, store.dir_for(project_id, "plan")) if plan else None,
        previous_script_ids=tuple(s.id for s in built_from.segments) if built_from else None,
    )


def adopt_generated_plan(
    store: DocumentStore, project_id: str, built: EditPlan, artifact_dir: Path
) -> EditPlan:
    """Store a built plan as the project's next version. It builds on `built.parent_version`;
    if a person saved another version meanwhile, nothing is stored and the job fails (the build
    stays cached). A build identical to the current version is not stored twice. The audio files
    are copied next to the versions (names are content hashes, so versions share them)."""
    head = store.read(project_id, "plan", EditPlan)
    if head is not None and same_content(head, built):
        return head
    audio_dir = store.dir_for(project_id, "plan")
    for seg in built.segments:
        if seg.audio is not None:
            dst = audio_dir / seg.audio.file
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(artifact_dir / seg.audio.file, dst)

    def build(_doc_id: str, version: int, parent: int | None) -> EditPlan:
        # The id is the one the build gave (derived from the project), not a fresh one: stored
        # and built plan are then the same bytes.
        return built.model_copy(
            update={"project_id": project_id, "version": version, "parent_version": parent}
        )

    try:
        return store.append(project_id, "plan", built.parent_version, build, author="ai")
    except StaleBase as e:
        raise Conflict(
            f"the plan changed while it was being built (it was version {built.parent_version}, "
            f"now {e.current}); build again to build on the current one"
        ) from e
