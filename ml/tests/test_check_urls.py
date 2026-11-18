"""Tests for scripts/check_urls.py (the local-only URL check run by make check)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_urls.py"


@pytest.fixture(scope="module")
def check_urls() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_urls", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("text", "bad"),
    [
        ('url = "http://localhost:11434/api/chat"', []),
        ("fetch(`http://127.0.0.1:8000/sessions`)", []),
        ("x = 'http://[::1]:8000'", []),
        ('u = "https://api.anthropic.com/v1"', ["https://api.anthropic.com/v1"]),
        ("see http://example.com and http://localhost", ["http://example.com"]),
        ("ws://evil.io/socket", ["ws://evil.io/socket"]),
        ("no urls here", []),
    ],
)
def test_offending_urls(check_urls: ModuleType, text: str, bad: list[str]) -> None:
    assert check_urls.offending_urls(text) == bad


def test_scan_finds_runtime_files_only(check_urls: ModuleType, tmp_path: Path) -> None:
    pkg = tmp_path / "ml" / "bowlsense"
    pkg.mkdir(parents=True)
    (pkg / "a.py").write_text('A = "http://localhost:1"\nB = "https://remote.example"\n')
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "dl.py").write_text('U = "https://weights.example"\n')
    assert check_urls.scan(tmp_path) == ["ml/bowlsense/a.py:2: https://remote.example"]


def test_repo_is_clean(check_urls: ModuleType) -> None:
    assert check_urls.scan() == []
