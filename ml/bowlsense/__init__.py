"""BowlSense: AI analysis of cricket bowling from video and images. Runs 100% locally."""

from __future__ import annotations

import os

# Telemetry off for every library before any of them is imported.
for _var, _value in (
    ("YOLO_OFFLINE", "1"),
    ("ULTRALYTICS_SYNC", "False"),
    ("HF_HUB_DISABLE_TELEMETRY", "1"),
    ("DO_NOT_TRACK", "1"),
    ("NEXT_TELEMETRY_DISABLED", "1"),
):
    os.environ.setdefault(_var, _value)

from bowlsense import offline  # noqa: E402

offline.enable_if_requested()

__version__ = "0.1.0"
