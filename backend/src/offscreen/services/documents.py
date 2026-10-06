"""Versioned documents: save with `base_version`, read any version, compare, restore.

Only the script has an editor so far (M4); the plan joins in M5 through the same store."""

from __future__ import annotations

from pydantic import ValidationError

from offscreen.algo.docdiff import diff_documents
from offscreen.config import AppConfig
from offscreen.domain.document import DocumentDiff, DocumentVersion
from offscreen.domain.script import Author, Script, ScriptContent
from offscreen.services.errors import Conflict, InvalidInput, NotFound
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
