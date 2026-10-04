"""Versioned schema migrations for offscreen.db.

The applied version lives in SQLite's `PRAGMA user_version`. To change the schema, append
a `Migration` with the next version number; never edit one that has shipped. Each migration
runs in a single transaction together with its version bump, so a failure leaves the
database at the previous version.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Migration:
    version: int
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        (
            """
            CREATE TABLE assets (
                id TEXT NOT NULL PRIMARY KEY,
                title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                probe_json TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
            """,
            "CREATE UNIQUE INDEX ix_assets_fingerprint ON assets (fingerprint)",
            """
            CREATE TABLE artifacts (
                cache_key TEXT NOT NULL PRIMARY KEY,
                stage TEXT NOT NULL,
                stage_version INTEGER NOT NULL,
                scope TEXT NOT NULL,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                created_at DATETIME NOT NULL,
                last_used_at DATETIME NOT NULL
            )
            """,
            "CREATE INDEX ix_artifacts_stage ON artifacts (stage)",
            "CREATE INDEX ix_artifacts_last_used_at ON artifacts (last_used_at)",
        ),
    ),
    Migration(
        2,
        (
            """
            CREATE TABLE llm_calls (
                id TEXT NOT NULL PRIMARY KEY,
                job_id TEXT,
                task TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                retries INTEGER NOT NULL,
                in_tokens INTEGER NOT NULL,
                out_tokens INTEGER NOT NULL,
                cached_tokens INTEGER NOT NULL,
                cost_usd FLOAT NOT NULL,
                latency_ms INTEGER NOT NULL,
                req_path TEXT,
                resp_path TEXT,
                created_at DATETIME NOT NULL
            )
            """,
            "CREATE INDEX ix_llm_calls_job_id ON llm_calls (job_id)",
            "CREATE INDEX ix_llm_calls_task ON llm_calls (task)",
            "CREATE INDEX ix_llm_calls_created_at ON llm_calls (created_at)",
        ),
    ),
    Migration(
        3,
        (
            """
            CREATE TABLE jobs (
                id TEXT NOT NULL PRIMARY KEY,
                stage TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                lane TEXT NOT NULL,
                status TEXT NOT NULL,
                progress FLOAT NOT NULL,
                message TEXT NOT NULL,
                cache_key TEXT,
                attempt INTEGER NOT NULL,
                error TEXT,
                log_path TEXT,
                cancel_requested BOOLEAN NOT NULL,
                not_before DATETIME,
                heartbeat_at DATETIME,
                created_at DATETIME NOT NULL,
                started_at DATETIME,
                finished_at DATETIME
            )
            """,
            "CREATE INDEX ix_jobs_stage ON jobs (stage)",
            "CREATE INDEX ix_jobs_lane ON jobs (lane)",
            "CREATE INDEX ix_jobs_status ON jobs (status)",
            "CREATE INDEX ix_jobs_created_at ON jobs (created_at)",
        ),
    ),
)

LATEST_VERSION = MIGRATIONS[-1].version


class MigrationError(RuntimeError):
    pass


def _check_sequence(migrations: tuple[Migration, ...]) -> None:
    if [m.version for m in migrations] != list(range(1, len(migrations) + 1)):
        raise MigrationError("migration versions must be 1..N with no gaps or repeats")


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def _enable_wal(conn: sqlite3.Connection, attempts: int = 100) -> None:
    """Switching the journal mode needs an exclusive lock that SQLite's busy handler does not
    always wait for, so racing first-time openers retry. WAL is persistent: once one process
    has set it, this is a no-op for everyone else."""
    for i in range(attempts):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or i == attempts - 1:
                raise
            time.sleep(0.05)


def migrate(path: Path, migrations: tuple[Migration, ...] = MIGRATIONS) -> int:
    """Bring the database at `path` up to the latest version (creating it if needed).
    Returns the resulting version. Safe to call from several processes at once."""
    _check_sequence(migrations)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)  # we manage BEGIN ourselves
    try:
        _enable_wal(conn)
        # IMMEDIATE takes the write lock first, so two processes cannot both see "v0".
        conn.execute("BEGIN IMMEDIATE")
        try:
            version = current_version(conn)
            if version > len(migrations):
                raise MigrationError(
                    f"database is at version {version}, newer than this code "
                    f"({len(migrations)}); upgrade the application"
                )
            for m in migrations[version:]:
                for stmt in m.statements:
                    conn.execute(stmt)
                conn.execute(f"PRAGMA user_version = {m.version}")  # int; not user input
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return current_version(conn)
    finally:
        conn.close()
