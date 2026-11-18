"""Tests that errors carry actionable messages."""

from __future__ import annotations

from bowlsense.errors import (
    BowlSenseError,
    LowFpsError,
    ModelDownloadError,
    NoBowlerDetectedError,
    NotFoundError,
    UnsupportedMediaError,
)


def test_low_fps_message() -> None:
    err = LowFpsError(30, 60)
    assert isinstance(err, BowlSenseError)
    assert "30.0 fps" in str(err) and "--allow-low-fps" in str(err)


def test_unsupported_media_lists_allowed() -> None:
    err = UnsupportedMediaError(".gif", (".mp4", ".png"))
    assert ".gif" in str(err) and ".mp4, .png" in str(err)


def test_no_bowler_mentions_override() -> None:
    assert "--bowler-track-id" in str(NoBowlerDetectedError("only 1 person seen"))


def test_model_download_says_where() -> None:
    msg = str(ModelDownloadError("rtm.onnx", "https://example.org/rtm.onnx", "/cache"))
    assert "rtm.onnx" in msg and "/cache" in msg


def test_not_found() -> None:
    assert "session" in str(NotFoundError("session", "abc"))
