"""Local SQLite database: ORM models, engine/session setup, Alembic migrations, repository."""

from __future__ import annotations

from bowlsense.config import Settings
from bowlsense.db.repo import (
    BowlerRecord,
    CoachMessageRecord,
    JobRecord,
    Repository,
    SessionRecord,
)
from bowlsense.db.session import get_engine

__all__ = [
    "BowlerRecord",
    "CoachMessageRecord",
    "JobRecord",
    "Repository",
    "SessionRecord",
    "open_repository",
]


def open_repository(settings: Settings) -> Repository:
    """Open (creating and migrating if needed) the database configured in ``settings``."""
    return Repository(get_engine(str(settings.storage.db_file)))
