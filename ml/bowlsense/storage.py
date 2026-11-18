"""Local file storage under ``data/storage/`` (``raw/`` uploads, ``outputs/<session_id>/``).

Uploaded file names are never used as paths: files are saved as ``<uuid>.<ext>``. Every path
coming from outside is resolved and checked to stay inside the storage root.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from bowlsense.errors import InvalidInputError, UnsupportedMediaError

CHUNK_BYTES = 1024 * 1024
RAW_DIR = "raw"
OUTPUTS_DIR = "outputs"


class PathTraversalError(InvalidInputError):
    """A requested path escapes the allowed storage folder."""


class UploadTooLargeError(InvalidInputError):
    """An upload exceeded the configured size limit."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(f"Upload is larger than the {max_bytes // (1024 * 1024)} MB limit.")


class LocalStorage:
    """File storage rooted at a local folder. All public paths are POSIX-style and relative."""

    def __init__(self, root: Path) -> None:
        """Create the storage folders under ``root`` if they don't exist."""
        self.root = root.resolve()
        (self.root / RAW_DIR).mkdir(parents=True, exist_ok=True)
        (self.root / OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)

    def resolve(self, rel_path: str, *, within: str | None = None) -> Path:
        """Turn a relative storage path into an absolute one, refusing traversal.

        Args:
            rel_path: POSIX path relative to the storage root, e.g. ``outputs/abc/overlay.mp4``.
            within: optional top-level folder the path must stay inside (``"outputs"``).

        Raises:
            PathTraversalError: if the path is absolute, contains ``..``, or escapes the root.
        """
        pure = PurePosixPath(rel_path.replace("\\", "/"))
        if not rel_path or pure.is_absolute() or ".." in pure.parts or ":" in rel_path:
            raise PathTraversalError(f"Invalid storage path '{rel_path}'.")
        base = self.root / within if within else self.root
        target = (self.root / Path(*pure.parts)).resolve()
        if not target.is_relative_to(base.resolve()):
            raise PathTraversalError(f"Path '{rel_path}' is outside the allowed folder.")
        return target

    def relative(self, path: Path) -> str:
        """Return ``path`` as a POSIX path relative to the storage root."""
        return path.resolve().relative_to(self.root).as_posix()

    def save_upload(
        self,
        stream: BinaryIO,
        original_name: str,
        allowed_extensions: Iterable[str],
        max_bytes: int,
    ) -> str:
        """Stream an upload to ``raw/<uuid>.<ext>`` in chunks and return its relative path.

        Only the extension of ``original_name`` is used. A partial file is removed on error.

        Raises:
            UnsupportedMediaError: if the extension is not allowed.
            UploadTooLargeError: if more than ``max_bytes`` are received.
        """
        allowed = tuple(e.lower() for e in allowed_extensions)
        suffix = PurePosixPath(original_name.replace("\\", "/")).suffix.lower()
        if suffix not in allowed:
            raise UnsupportedMediaError(suffix, allowed)
        rel = f"{RAW_DIR}/{uuid.uuid4()}{suffix}"
        target = self.resolve(rel, within=RAW_DIR)
        written = 0
        try:
            with target.open("wb") as out:
                while chunk := stream.read(CHUNK_BYTES):
                    written += len(chunk)
                    if written > max_bytes:
                        raise UploadTooLargeError(max_bytes)
                    out.write(chunk)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return rel

    def session_output_dir(self, session_id: str) -> Path:
        """Return (and create) ``outputs/<session_id>/``."""
        path = self.resolve(f"{OUTPUTS_DIR}/{session_id}", within=OUTPUTS_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def delete_session_files(self, session_id: str, video_path: str | None) -> None:
        """Delete a session's raw video and its outputs folder (missing files are fine)."""
        if video_path:
            self.resolve(video_path, within=RAW_DIR).unlink(missing_ok=True)
        out = self.resolve(f"{OUTPUTS_DIR}/{session_id}", within=OUTPUTS_DIR)
        if out.exists():
            shutil.rmtree(out)

    def wipe(self) -> None:
        """Delete every stored file, keeping the empty ``raw/`` and ``outputs/`` folders."""
        for sub in (RAW_DIR, OUTPUTS_DIR):
            folder = self.root / sub
            for child in folder.iterdir():
                if child.name == ".gitkeep":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    def usage_bytes(self) -> int:
        """Total size of stored files in bytes."""
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
