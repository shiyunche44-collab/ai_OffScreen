from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlmodel import SQLModel

from offscreen.domain.artifact import Manifest
from offscreen.domain.asset import DerivedFiles, MediaAsset
from offscreen.store import models
from offscreen.store.db import Database
from offscreen.store.migrations import LATEST_VERSION, Migration, MigrationError, migrate
from offscreen.store.repos import ArtifactIndex, AssetRepo

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "domain"


def fixture(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "data" / "offscreen.db")
    yield d  # type: ignore[misc]
    d.close()


# --- migrations -------------------------------------------------------------------------


def test_migrate_creates_database_at_latest_version_in_wal_mode(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "offscreen.db"
    assert migrate(p) == LATEST_VERSION
    assert migrate(p) == LATEST_VERSION  # idempotent
    with sqlite3.connect(p) as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"assets", "artifacts"} <= tables


def test_later_migrations_apply_on_top_of_existing_data(tmp_path: Path) -> None:
    p = tmp_path / "x.db"
    v1 = Migration(1, ("CREATE TABLE t (a INTEGER)",))
    v2 = Migration(2, ("ALTER TABLE t ADD COLUMN b TEXT DEFAULT 'x'",))
    assert migrate(p, (v1,)) == 1
    with sqlite3.connect(p) as c:
        c.execute("INSERT INTO t (a) VALUES (7)")
    assert migrate(p, (v1, v2)) == 2
    with sqlite3.connect(p) as c:
        assert c.execute("SELECT a, b FROM t").fetchall() == [(7, "x")]


def test_failed_migration_rolls_back_including_version(tmp_path: Path) -> None:
    p = tmp_path / "x.db"
    v1 = Migration(1, ("CREATE TABLE t (a INTEGER)",))
    bad = Migration(2, ("CREATE TABLE u (a INTEGER)", "THIS IS NOT SQL"))
    migrate(p, (v1,))
    with pytest.raises(sqlite3.Error):
        migrate(p, (v1, bad))
    with sqlite3.connect(p) as c:
        assert c.execute("PRAGMA user_version").fetchone()[0] == 1
        names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "u" not in names


def test_database_newer_than_code_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "x.db"
    migrate(p, (Migration(1, ("CREATE TABLE t (a INTEGER)",)),))
    with pytest.raises(MigrationError, match="newer"):
        migrate(p, ())  # code that knows no migrations at all
    with sqlite3.connect(p) as c:
        c.execute("PRAGMA user_version = 99")
    with pytest.raises(MigrationError, match="newer"):
        migrate(p)


def test_migration_numbers_must_be_contiguous(tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="no gaps"):
        migrate(tmp_path / "x.db", (Migration(2, ("SELECT 1",)),))


def test_concurrent_first_open_is_safe(tmp_path: Path) -> None:
    p = tmp_path / "x.db"
    errors: list[BaseException] = []

    def go() -> None:
        try:
            assert migrate(p) == LATEST_VERSION
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []


def test_models_match_migrated_schema(tmp_path: Path) -> None:
    """Guards against editing models.py without writing a migration (or vice versa)."""
    migrate(tmp_path / "m.db")
    migrated = inspect(create_engine(f"sqlite:///{tmp_path / 'm.db'}"))
    fresh = create_engine("sqlite://")
    SQLModel.metadata.create_all(fresh)
    expected = inspect(fresh)

    assert set(migrated.get_table_names()) == set(expected.get_table_names())
    for table in expected.get_table_names():

        def cols(i, t=table):  # type: ignore[no-untyped-def]
            return {
                c["name"]: (
                    str(c["type"]).replace("VARCHAR", "TEXT"),
                    c["nullable"],
                    bool(c["primary_key"]),
                )
                for c in i.get_columns(t)
            }

        def idx(i, t=table):  # type: ignore[no-untyped-def]
            return {(tuple(x["column_names"]), bool(x["unique"])) for x in i.get_indexes(t)}

        assert cols(migrated) == cols(expected), table
        assert idx(migrated) == idx(expected), table
    assert models.AssetRow and models.ArtifactRow


# --- database ---------------------------------------------------------------------------


def test_connections_use_wal_foreign_keys_and_busy_timeout(db: Database) -> None:
    with db.engine.connect() as c:
        get = lambda p: c.exec_driver_sql(f"PRAGMA {p}").scalar()  # noqa: E731
        assert get("journal_mode") == "wal"
        assert get("foreign_keys") == 1
        assert get("busy_timeout") == 5000


def test_session_rolls_back_on_error(db: Database) -> None:
    repo = AssetRepo(db)
    asset = MediaAsset.model_validate(fixture("asset"))
    with pytest.raises(RuntimeError), db.session() as s:
        s.add(
            models.AssetRow(
                id=asset.id, title="t", source_path="p", fingerprint="f", probe_json="{}"
            )
        )
        raise RuntimeError
    assert repo.get(asset.id) is None


# --- repositories -----------------------------------------------------------------------


def test_asset_add_get_list_and_idempotent_by_fingerprint(db: Database) -> None:
    repo = AssetRepo(db)
    asset = MediaAsset.model_validate(fixture("asset"))
    assert repo.add(asset) == asset
    assert repo.get(asset.id) == asset
    assert repo.find_by_fingerprint(asset.fingerprint) == asset
    assert repo.list() == [asset]

    twin = asset.model_copy(update={"id": "ast_second", "title": "same file again"})
    assert repo.add(twin) == asset  # returns the one already registered
    assert len(repo.list()) == 1
    assert repo.get("ast_nope") is None


def test_asset_update_persists_derived_files(db: Database) -> None:
    repo = AssetRepo(db)
    asset = repo.add(MediaAsset.model_validate(fixture("asset")))
    updated = asset.model_copy(update={"derived": DerivedFiles(proxy="assets/x/proxy_540p.mp4")})
    repo.update(updated)
    assert repo.get(asset.id) == updated
    with pytest.raises(KeyError):
        repo.update(updated.model_copy(update={"id": "ast_missing"}))


def test_artifact_index_record_touch_list_delete(db: Database) -> None:
    idx = ArtifactIndex(db)
    m = Manifest.model_validate(fixture("manifest"))
    row = idx.record(m, "artifacts/analysis.shots/abab")
    assert (row.stage, row.size, row.content_hash) == (
        "analysis.shots",
        2048 + 18342,
        m.content_hash(),
    )
    assert json.loads(row.scope) == {"asset_id": "ast_sintel01"}

    other = m.model_copy(update={"stage": "analysis.proxy", "cache_key": "sha256:" + "c" * 64})
    idx.record(other, "artifacts/analysis.proxy/cccc")
    assert [r.stage for r in idx.list()] == ["analysis.shots", "analysis.proxy"]
    assert [r.cache_key for r in idx.list("analysis.proxy")] == [other.cache_key]

    before = idx.get(m.cache_key).last_used_at  # type: ignore[union-attr]
    idx.touch(m.cache_key)
    idx.touch("sha256:" + "0" * 64)  # unknown key: no-op
    assert idx.get(m.cache_key).last_used_at >= before  # type: ignore[union-attr]
    assert [r.stage for r in idx.list()] == ["analysis.proxy", "analysis.shots"]  # LRU order

    idx.record(m, "artifacts/analysis.shots/moved")  # upsert, not duplicate
    assert len(idx.list("analysis.shots")) == 1
    assert idx.get(m.cache_key).path == "artifacts/analysis.shots/moved"  # type: ignore[union-attr]

    idx.delete(m.cache_key)
    assert idx.get(m.cache_key) is None
