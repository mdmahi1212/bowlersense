"""Tests for local file storage."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from bowlsense.errors import UnsupportedMediaError
from bowlsense.storage import LocalStorage, PathTraversalError, UploadTooLargeError

ALLOWED = (".mp4", ".png")


def test_creates_folders(tmp_path: Path) -> None:
    st = LocalStorage(tmp_path / "s")
    assert (tmp_path / "s" / "raw").is_dir() and (tmp_path / "s" / "outputs").is_dir()
    assert st.usage_bytes() == 0


def test_save_upload_uses_uuid_name(storage: LocalStorage) -> None:
    rel = storage.save_upload(io.BytesIO(b"abc" * 1000), "../../evil name.MP4", ALLOWED, 10_000)
    assert rel.startswith("raw/") and rel.endswith(".mp4")
    assert "evil" not in rel
    assert storage.resolve(rel).read_bytes() == b"abc" * 1000
    assert storage.relative(storage.resolve(rel)) == rel


def test_save_upload_rejects_type(storage: LocalStorage) -> None:
    with pytest.raises(UnsupportedMediaError):
        storage.save_upload(io.BytesIO(b"x"), "clip.gif", ALLOWED, 100)
    with pytest.raises(UnsupportedMediaError):
        storage.save_upload(io.BytesIO(b"x"), "noext", ALLOWED, 100)


def test_save_upload_too_large_cleans_up(storage: LocalStorage) -> None:
    with pytest.raises(UploadTooLargeError):
        storage.save_upload(io.BytesIO(b"x" * 3_000_000), "a.mp4", ALLOWED, 1_000_000)
    assert list((storage.root / "raw").iterdir()) == []


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "../secret.txt",
        "outputs/../../x",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "outputs\\..\\..\\x",
    ],
)
def test_resolve_rejects_traversal(storage: LocalStorage, bad: str) -> None:
    with pytest.raises(PathTraversalError):
        storage.resolve(bad)


def test_resolve_within(storage: LocalStorage) -> None:
    assert storage.resolve("outputs/a/b.png", within="outputs").name == "b.png"
    with pytest.raises(PathTraversalError):
        storage.resolve("raw/a.mp4", within="outputs")


def test_session_files_and_wipe(storage: LocalStorage) -> None:
    out = storage.session_output_dir("s1")
    (out / "k.png").write_bytes(b"12345")
    (storage.root / "raw" / ".gitkeep").write_bytes(b"")
    (storage.root / "raw" / "v.mp4").write_bytes(b"1")
    assert storage.usage_bytes() == 6
    storage.delete_session_files("s1", "raw/v.mp4")
    assert not out.exists() and not (storage.root / "raw" / "v.mp4").exists()
    storage.delete_session_files("s1", None)  # already gone: no error
    storage.session_output_dir("s2")
    (storage.root / "raw" / "w.mp4").write_bytes(b"1")
    storage.wipe()
    assert [p.name for p in (storage.root / "raw").iterdir()] == [".gitkeep"]
    assert list((storage.root / "outputs").iterdir()) == []
