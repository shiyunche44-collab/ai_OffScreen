"""SQLite connection management: WAL, foreign keys, busy timeout, migrations on open."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, create_engine

from offscreen.store.migrations import migrate

BUSY_TIMEOUT_MS = 5000


class Database:
    """One SQLite file shared by the API and Worker processes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.version = migrate(path)
        self.engine: Engine = create_engine(
            f"sqlite:///{path}", connect_args={"check_same_thread": False}
        )
        event.listen(self.engine, "connect", _set_pragmas)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A transaction: commits on success, rolls back on error. Rows read inside stay
        usable after the block (no expiry on commit)."""
        with Session(self.engine, expire_on_commit=False) as s:
            try:
                yield s
                s.commit()
            except BaseException:
                s.rollback()
                raise

    def close(self) -> None:
        self.engine.dispose()


def _set_pragmas(dbapi_conn: Any, _record: Any) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    cur.close()
