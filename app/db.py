"""SQLite persistence with serialized writers and snapshot reads.

There is deliberately only *one* connection factory: batch readings and the
batch audit live in the same database file and are written inside the same
transaction, so an accept can never commit without its audit row and a
rejected batch can never leave readings behind.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "traffic.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    sample_id   INTEGER PRIMARY KEY,
    local_time  TEXT    NOT NULL,
    pass_count  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_ref   TEXT    NOT NULL,
    status      TEXT    NOT NULL CHECK (status IN ('accepted', 'rejected')),
    reason      TEXT,
    record_count INTEGER NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


def get_db_path() -> Path:
    return DB_PATH


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a short-lived connection (autocommit mode, WAL, busy timeout).

    ``check_same_thread=False`` keeps the connection usable from the request
    handler thread that owns it; callers must not share one connection across
    threads concurrently.  Each request opens its own connection instead.
    """

    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        timeout=30,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE: acquires the write lock up front.

    Only one writer exists at any moment, so two concurrent batches can never
    interleave their validation/insertion and accept mutually conflicting data.
    Contenders block on busy_timeout and then observe the winner's committed
    state when re-validating.  The audit row is written inside this same
    transaction, which makes every batch decision atomic.
    """

    conn.execute("BEGIN IMMEDIATE;")
    try:
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise


@contextmanager
def read_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Single-snapshot read; a query never mixes old verdicts with new rows."""

    conn.execute("BEGIN;")
    try:
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
