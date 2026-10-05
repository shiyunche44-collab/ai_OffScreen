from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from offscreen.domain.asset import MediaAsset
from offscreen.domain.project import ProjectOptions
from offscreen.store.db import Database
from offscreen.store.repos import AssetRepo, ProjectRepo

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "domain"


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "offscreen.db")
    yield d  # type: ignore[misc]
    d.close()


@pytest.fixture
def asset_id(db: Database) -> str:
    asset = MediaAsset.model_validate(json.loads((FIXTURES / "asset.json").read_text("utf-8")))
    return AssetRepo(db).add(asset).id


def test_add_get_list_roundtrip(db: Database, asset_id: str) -> None:
    repo = ProjectRepo(db)
    opts = ProjectOptions(minutes=2.5, voice="v1", style="funny", spoil_ending=False)
    a = repo.add(asset_id, "first", opts)
    b = repo.add(asset_id, "second", ProjectOptions())

    assert a.id.startswith("prj_") and a.asset_id == asset_id
    assert repo.get(a.id) == a
    assert repo.get(a.id).options == opts  # type: ignore[union-attr]
    assert [p.id for p in repo.list()] == [b.id, a.id]  # newest first
    assert repo.get("prj_missing") is None


def test_a_project_needs_an_existing_asset(db: Database) -> None:
    with pytest.raises(IntegrityError):
        ProjectRepo(db).add("ast_missing", "orphan", ProjectOptions())


@pytest.mark.parametrize(
    "bad",
    [{"minutes": 0}, {"minutes": -1}, {"minutes": 1000}, {"style": ""}, {"voice": ""}, {"x": 1}],
)
def test_options_are_validated(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ProjectOptions(**bad)  # type: ignore[arg-type]
