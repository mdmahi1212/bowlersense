"""SQLite engine, connection pragmas, session factory and Alembic migration runner."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import sessionmaker

from bowlsense.errors import DatabaseError

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
BUSY_TIMEOUT_MS = 30_000
IMMEDIATE = "sqlite_begin_immediate"


def _set_pragmas(dbapi_conn: sqlite3.Connection, _record: Any) -> None:
    # Take control of transactions so we can issue BEGIN IMMEDIATE ourselves.
    dbapi_conn.isolation_level = None
    cur = dbapi_conn.cursor()
    try:
        cur.execute("PRAGMA foreign_keys = ON")
        cur.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        cur.execute("PRAGMA journal_mode = WAL")
        cur.execute("PRAGMA synchronous = NORMAL")
    finally:
        cur.close()


def _begin(conn: Connection) -> None:
    # A write lock taken up front avoids "database is locked" upgrade deadlocks when the
    # API and the worker write at the same time.
    if conn.get_execution_options().get(IMMEDIATE):
        conn.exec_driver_sql("BEGIN IMMEDIATE")
    else:
        conn.exec_driver_sql("BEGIN")


def make_engine(db_path: Path | str) -> Engine:
    """Create an engine for the SQLite file at ``db_path`` (parent folders are created).

    Every connection gets ``foreign_keys=ON``, WAL journaling and a busy timeout.
    """
    path = Path(db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatabaseError(f"Cannot create database folder {path.parent}: {exc}") from exc
    engine = create_engine(f"sqlite:///{path.as_posix()}", future=True)
    event.listen(engine, "connect", _set_pragmas)
    event.listen(engine, "begin", _begin)
    return engine


def immediate(engine: Engine) -> Engine:
    """Return a view of ``engine`` whose transactions start with ``BEGIN IMMEDIATE``."""
    return engine.execution_options(**{IMMEDIATE: True})


def make_sessionmaker(engine: Engine) -> sessionmaker[Any]:
    """Return a session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def alembic_config(engine: Engine) -> AlembicConfig:
    """Build an Alembic config pointing at the bundled migrations and this engine."""
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    # The engine is handed over directly (no URL in the config): Windows drive letters in a
    # URL would break Alembic's ini-style interpolation.
    cfg.attributes["engine"] = engine
    return cfg


def upgrade(engine: Engine, revision: str = "head") -> None:
    """Apply Alembic migrations up to ``revision`` (default: latest)."""
    try:
        command.upgrade(alembic_config(engine), revision)
    except Exception as exc:
        raise DatabaseError(f"Database migration to '{revision}' failed: {exc}") from exc


@lru_cache(maxsize=8)
def get_engine(db_path: str) -> Engine:
    """Return a cached, migrated engine for ``db_path`` (one per process per file)."""
    engine = make_engine(db_path)
    upgrade(engine)
    return engine
