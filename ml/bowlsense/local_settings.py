"""User-editable local settings (the web app's settings page), stored next to the database.

They override a few config defaults: the Ollama model, the pose backend, GPU use and whether
the LLM coach is used at all. The file is plain JSON at ``<db folder>/settings.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bowlsense.config import Settings
from bowlsense.device import set_gpu_allowed

log = logging.getLogger(__name__)


class LocalSettings(BaseModel):
    """What the settings page can change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    coach_model: str = Field(min_length=1, max_length=100)
    pose_backend: Literal["rtm", "mediapipe"]
    use_gpu: bool = True
    use_llm: bool = True


def settings_file(settings: Settings) -> Path:
    """Where the local settings live."""
    return settings.storage.db_file.parent / "settings.json"


def defaults(settings: Settings) -> LocalSettings:
    """Local settings taken from the config file."""
    return LocalSettings(coach_model=settings.coach.model, pose_backend=settings.pose.backend)


def load_local_settings(settings: Settings) -> LocalSettings:
    """The saved local settings, or the config defaults (an unreadable file is ignored)."""
    path = settings_file(settings)
    if not path.is_file():
        return defaults(settings)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        return LocalSettings.model_validate({**defaults(settings).model_dump(), **saved})
    except (OSError, ValueError, ValidationError) as exc:
        log.warning("ignoring unreadable %s: %s", path, exc)
        return defaults(settings)


def save_local_settings(settings: Settings, local: LocalSettings) -> LocalSettings:
    """Write the local settings file."""
    path = settings_file(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(local.model_dump_json(indent=1), encoding="utf-8")
    return local


def effective_settings(settings: Settings, local: LocalSettings) -> Settings:
    """``settings`` with the local overrides applied (and GPU use switched accordingly)."""
    set_gpu_allowed(local.use_gpu)
    return settings.model_copy(
        update={
            "coach": settings.coach.model_copy(update={"model": local.coach_model}),
            "pose": settings.pose.model_copy(update={"backend": local.pose_backend}),
        }
    )
