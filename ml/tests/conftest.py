"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from bowlsense import offline
from bowlsense.config import Settings, load_settings
from bowlsense.db.repo import Repository
from bowlsense.db.session import make_engine, upgrade
from bowlsense.storage import LocalStorage

REAL_DATA_DIR = Path(__file__).parent / "data" / "real"


@pytest.fixture(scope="session", autouse=True)
def offline_when_requested() -> Iterator[None]:
    """With ``BOWLSENSE_OFFLINE=1`` the whole test session runs behind the network guard."""
    offline.enable_if_requested()
    yield


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Default settings from config/default.yaml."""
    return load_settings()


@pytest.fixture()
def repo(tmp_path: Path) -> Iterator[Repository]:
    """A fresh, migrated SQLite database per test."""
    engine = make_engine(tmp_path / "test.db")
    upgrade(engine)
    yield Repository(engine)
    engine.dispose()


@pytest.fixture()
def storage(tmp_path: Path) -> LocalStorage:
    """A fresh storage root per test."""
    return LocalStorage(tmp_path / "storage")


@pytest.fixture()
def no_offline_guard() -> Iterator[None]:
    """Make sure the offline guard is off for the test and restored afterwards."""
    was_on = offline.is_enabled()
    offline.disable()
    yield
    offline.disable()
    if was_on:
        offline.enable()


def real_clips() -> list[Path]:
    """Real video clips placed in tests/data/real (optional; tests skip when empty)."""
    exts = {".mp4", ".mov", ".avi"}
    if not REAL_DATA_DIR.is_dir():
        return []
    return sorted(p for p in REAL_DATA_DIR.iterdir() if p.suffix.lower() in exts)
