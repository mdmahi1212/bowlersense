"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from bowlsense.config import Settings, load_settings
from bowlsense.db import SQLiteDatabase

REAL_DATA_DIR = Path(__file__).parent / "data" / "real"


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Default settings from config/default.yaml."""
    return load_settings()


@pytest.fixture()
def db(tmp_path: Path) -> Iterator[SQLiteDatabase]:
    """A fresh on-disk SQLite database per test."""
    database = SQLiteDatabase(tmp_path / "test.db")
    yield database
    database.close()


def real_clips() -> list[Path]:
    """Real video clips placed in tests/data/real (optional; tests skip when empty)."""
    exts = {".mp4", ".mov", ".avi"}
    if not REAL_DATA_DIR.is_dir():
        return []
    return sorted(p for p in REAL_DATA_DIR.iterdir() if p.suffix.lower() in exts)
