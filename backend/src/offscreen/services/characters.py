"""The characters of a movie as people see them: the AI's output with human edits applied.

Reading merges the revision layer in (`algo.characters.apply_overrides`); editing writes to it.
Edits carry the face centre of the person they are about, so they survive a re-clustering
(which renumbers people): every read and every edit first re-addresses them to the current
characters, and what no longer matches anyone is kept aside, never dropped."""

from __future__ import annotations

import threading

import numpy as np
from pydantic import BaseModel

from offscreen.algo.characters import apply_overrides, remap_overrides
from offscreen.config import AppConfig
from offscreen.domain.index import (
    Character,
    CharacterOverride,
    CharacterOverrides,
    Characters,
    FaceCluster,
)
from offscreen.engine import Artifact
from offscreen.services.errors import InvalidInput, NotFound
from offscreen.services.jobs import JobService
from offscreen.services.pipeline import Pipeline
from offscreen.stages.analysis.characters import CENTROIDS_FILE, CHARACTERS_FILE
from offscreen.store.db import Database
from offscreen.store.overrides import OverridesStore
from offscreen.store.repos import AssetRepo


class CharactersView(BaseModel):
    asset_id: str
    characters: list[Character]
    """Edits applied; ignored and merged-away characters left out. Thumbnails are paths relative
    to the data directory."""
    merged: dict[str, str]
    """`merged-away id -> the character it joined` (faces and cast still use the old ids)."""
    ignored: list[str]
    unmatched_edits: int
    """Edits whose person no longer exists after a re-clustering."""
    named: bool
    """The names come from the naming stage (False: only clusters, nobody is named yet)."""


class CharacterEdit(BaseModel):
    """What to change about one character; only the fields present are touched."""

    name: str | None = None
    """The person's name; an empty string says the character has none (the AI was wrong)."""
    aliases: list[str] | None = None
    ignored: bool | None = None
    merged_into: str | None = None
    """Join this character to another one (they are the same person); null: leave it."""
    reset: bool = False
    """Drop every edit made to this character."""


class CharacterService:
    def __init__(self, cfg: AppConfig, db: Database, jobs: JobService) -> None:
        self.cfg = cfg
        self.db = db
        self.jobs = jobs
        self.assets = AssetRepo(db)
        self.store = OverridesStore(cfg.data_dir)
        self._lock = threading.Lock()

    def view(self, asset_id: str) -> CharactersView:
        self._require_asset(asset_id)
        with self._lock:
            doc, centres, art, named = self._current(asset_id)
            remapped, orphans = remap_overrides(
                self.store.read_characters(asset_id).overrides, centres
            )
        return self._view(asset_id, doc, art, named, remapped, len(orphans))

    def edit(self, asset_id: str, character_id: str, change: CharacterEdit) -> CharactersView:
        self._require_asset(asset_id)
        with self._lock:
            doc, centres, art, named = self._current(asset_id)
            if character_id not in centres:
                raise NotFound(f"unknown character {character_id}")
            target = change.merged_into
            if target is not None and (target == character_id or target not in centres):
                raise InvalidInput(f"cannot merge {character_id} into {target}")
            remapped, orphans = remap_overrides(
                self.store.read_characters(asset_id).overrides, centres
            )
            mine = next((o for o in remapped if o.character_id == character_id), None)
            rest = [o for o in remapped if o.character_id != character_id]
            if not change.reset:
                o = mine or CharacterOverride(
                    character_id=character_id, centroid=list(map(float, centres[character_id]))
                )
                fields = change.model_fields_set
                if "name" in fields:
                    o = o.model_copy(update={"name": change.name})
                if "aliases" in fields:
                    o = o.model_copy(update={"aliases": change.aliases})
                if "ignored" in fields and change.ignored is not None:
                    o = o.model_copy(update={"ignored": change.ignored})
                if "merged_into" in fields:
                    o = o.model_copy(
                        update={
                            "merged_into": target,
                            "merged_into_centroid": list(map(float, centres[target]))
                            if target
                            else None,
                        }
                    )
                rest.append(o)
            kept = [*rest, *orphans]
            self.store.write_characters(CharacterOverrides(asset_id=asset_id, overrides=kept))
        return self._view(asset_id, doc, art, named, rest, len(orphans))

    # ---- internals -------------------------------------------------------------------------
    def _current(self, asset_id: str) -> tuple[Characters, dict[str, list[float]], Artifact, bool]:
        """The newest characters document (named if the naming stage has run), their face
        centres by id, and where they came from."""
        with Pipeline(self.cfg, self.jobs.providers, db=self.db) as p:
            art = p.peek("analysis.naming", asset_id)
            named = art is not None
            if art is None:
                art = p.peek("analysis.characters", asset_id)
        if art is None:
            raise NotFound("analysis.characters has not been built yet")
        doc = art.read_model(CHARACTERS_FILE, Characters)
        matrix = np.load(art.path(CENTROIDS_FILE))
        centres: dict[str, list[float]] = {}
        for c in doc.characters:
            ref = c.face_cluster.centroid_ref if c.face_cluster else None
            if ref and "#" in ref and int(ref.split("#")[1]) < len(matrix):
                centres[c.id] = [float(x) for x in matrix[int(ref.split("#")[1])]]
            else:
                raise InvalidInput(f"character {c.id} has no face centre")
        return doc, centres, art, named

    def _view(
        self,
        asset_id: str,
        doc: Characters,
        art: Artifact,
        named: bool,
        overrides: list[CharacterOverride],
        unmatched: int,
    ) -> CharactersView:
        characters, merged, ignored = apply_overrides(doc.characters, overrides)
        base = self.cfg.data_dir.resolve()
        shown = [
            c.model_copy(
                update={
                    "face_cluster": FaceCluster(
                        size=c.face_cluster.size,
                        centroid_ref=c.face_cluster.centroid_ref,
                        thumbnails=[
                            art.path(t).resolve().relative_to(base).as_posix()
                            for t in c.face_cluster.thumbnails
                        ],
                    )
                }
            )
            if c.face_cluster
            else c
            for c in characters
        ]
        return CharactersView(
            asset_id=asset_id,
            characters=shown,
            merged=merged,
            ignored=ignored,
            unmatched_edits=unmatched,
            named=named,
        )

    def _require_asset(self, asset_id: str) -> None:
        if self.assets.get(asset_id) is None:
            raise NotFound(f"unknown asset {asset_id}")
