"""H.264 MP4 writing through the bundled ffmpeg (imageio-ffmpeg), browser-ready.

Raw BGR frames are streamed into ffmpeg's libx264 encoder (``-movflags +faststart`` so the
browser can start playing before the whole file has loaded; ``yuv420p`` for compatibility).
Nothing is held in memory beyond the current frame.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from bowlsense.errors import BowlSenseError
from bowlsense.io.video import Frame


class RenderError(BowlSenseError):
    """Video encoding failed."""


def ffmpeg_exe() -> str:
    """Path to the ffmpeg binary bundled with imageio-ffmpeg.

    Raises:
        RenderError: if it isn't available.
    """
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - it's a pinned dependency
        raise RenderError("imageio-ffmpeg is not installed; reinstall the requirements.") from exc
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not Path(exe).is_file():  # pragma: no cover - the wheel ships the binary
        raise RenderError(f"ffmpeg binary not found at {exe}.")
    return str(exe)


def even_size(width: int, height: int) -> tuple[int, int]:
    """yuv420p needs even dimensions; frames are padded up by at most one pixel."""
    return width + (width % 2), height + (height % 2)


class H264Writer:
    """Context manager that streams BGR frames of a fixed size into an H.264 MP4."""

    def __init__(
        self, path: Path, width: int, height: int, fps: float, preset: str, crf: int
    ) -> None:
        """Start ffmpeg; frames must then be exactly ``width`` x ``height``."""
        if width <= 0 or height <= 0 or fps <= 0:
            raise RenderError(f"Bad video size/fps: {width}x{height} @ {fps}.")
        self.path = path
        self.width, self.height = width, height
        self.out_w, self.out_h = even_size(width, height)
        self.frames = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        # ffmpeg's stderr goes to a file (a pipe could fill up and block); it lives as long
        # as the encoder and is closed in close()/abort().
        self._log = tempfile.TemporaryFile()  # noqa: SIM115
        cmd = [
            ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.out_w}x{self.out_h}", "-r", f"{fps:.6f}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]  # fmt: skip
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)

    def write(self, frame: Frame) -> None:
        """Encode one frame."""
        if frame.shape[:2] != (self.height, self.width) or frame.ndim != 3:
            raise RenderError(
                f"Frame is {frame.shape[1]}x{frame.shape[0]} but the video is "
                f"{self.width}x{self.height}."
            )
        if (self.out_w, self.out_h) != (self.width, self.height):
            frame = np.asarray(
                cv2.copyMakeBorder(
                    frame, 0, self.out_h - self.height, 0, self.out_w - self.width,
                    cv2.BORDER_REPLICATE,
                ),
                dtype=np.uint8,
            )  # fmt: skip
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except BrokenPipeError as exc:
            raise RenderError(f"ffmpeg stopped early: {self._stderr()}") from exc
        self.frames += 1

    def _stderr(self) -> str:
        self._log.seek(0)
        return self._log.read().decode("utf-8", "replace").strip()[-2000:]

    def close(self) -> None:
        """Finish encoding.

        Raises:
            RenderError: if ffmpeg failed or no frames were written.
        """
        try:
            if self._proc.stdin is not None and not self._proc.stdin.closed:
                self._proc.stdin.close()
            code = self._proc.wait(timeout=600)
            if code != 0:
                raise RenderError(f"ffmpeg failed (exit {code}): {self._stderr()}")
            if self.frames == 0:
                raise RenderError("No frames were written to the video.")
        finally:
            self._log.close()

    def abort(self) -> None:
        """Stop ffmpeg and delete the partial file."""
        try:
            self._proc.kill()
            self._proc.wait(timeout=30)
        finally:
            self._log.close()
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> H264Writer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def mp4_top_level_boxes(path: Path) -> list[str]:
    """Names of the top-level MP4 boxes in order (e.g. ftyp, moov, mdat). With faststart,
    ``moov`` (the index) comes before ``mdat`` (the media data)."""
    names: list[str] = []
    with path.open("rb") as f:
        size_total = path.stat().st_size
        pos = 0
        while pos + 8 <= size_total:
            f.seek(pos)
            header = f.read(8)
            size = int.from_bytes(header[:4], "big")
            name = header[4:8].decode("latin-1")
            if size == 1:
                size = int.from_bytes(f.read(8), "big")
            elif size == 0:
                size = size_total - pos
            if size < 8:
                break
            names.append(name)
            pos += size
    return names


def probe_codec(path: Path) -> str:
    """Video codec name ffmpeg reports for a file (e.g. "h264")."""
    out = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True
    ).stderr
    for line in out.splitlines():
        if "Video:" in line:
            return line.split("Video:", 1)[1].strip().split()[0].rstrip(",")
    raise RenderError(f"No video stream found in {path.name}.")
