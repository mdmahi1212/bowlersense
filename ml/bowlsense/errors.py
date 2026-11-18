"""BowlSense exception hierarchy.

Every error carries a message a user can act on (what went wrong and what to do about it).
"""

from __future__ import annotations


class BowlSenseError(Exception):
    """Base class for all BowlSense errors."""


class ConfigError(BowlSenseError):
    """The configuration file or an environment variable is missing or invalid."""


class InvalidInputError(BowlSenseError):
    """An input file or value is malformed."""


class UnsupportedMediaError(InvalidInputError):
    """The uploaded file type is not supported."""

    def __init__(self, suffix: str, allowed: tuple[str, ...]) -> None:
        self.suffix = suffix
        self.allowed = allowed
        super().__init__(
            f"Unsupported file type '{suffix or '(none)'}'. "
            f"Upload one of: {', '.join(allowed)}."
        )


class LowFpsError(BowlSenseError):
    """The video frame rate is too low for reliable analysis."""

    def __init__(self, fps: float, min_fps: float) -> None:
        self.fps = fps
        self.min_fps = min_fps
        super().__init__(
            f"Video is {fps:.1f} fps but at least {min_fps:.0f} fps is required. "
            "Record in slow-motion mode (120 fps or more), or pass --allow-low-fps "
            "to analyse anyway with reduced accuracy."
        )


class NoBowlerDetectedError(BowlSenseError):
    """No person could be identified as the bowler."""

    def __init__(self, detail: str = "") -> None:
        msg = (
            "No bowler was detected in the video. Make sure the full body is in frame "
            "and the bowler runs across the view, or pass --bowler-track-id."
        )
        super().__init__(f"{msg} {detail}".strip())


class PoseQualityError(BowlSenseError):
    """Pose keypoints are too unreliable to analyse."""


class PhaseDetectionError(BowlSenseError):
    """Delivery phases could not be located at all."""


class CalibrationError(BowlSenseError):
    """Pitch calibration points are missing or degenerate."""


class ModelDownloadError(BowlSenseError):
    """Model weights could not be downloaded."""

    def __init__(self, filename: str, url: str, target_dir: str) -> None:
        self.filename = filename
        super().__init__(
            f"Could not download model weights '{filename}'. If you are offline, download "
            f"it manually from {url} and place it in {target_dir}."
        )


class DatabaseError(BowlSenseError):
    """A database operation failed."""


class NotFoundError(DatabaseError):
    """A requested record does not exist."""

    def __init__(self, table: str, record_id: str) -> None:
        self.table = table
        self.record_id = record_id
        super().__init__(f"No {table} record with id '{record_id}'.")
