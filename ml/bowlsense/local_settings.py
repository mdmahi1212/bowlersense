"""User-editable local settings (the web app's settings page), stored next to the database.

They override a few config defaults: the Ollama model, the pose backend, GPU use, whether
the LLM coach is used at all, and whether the learned phase model is used. The file is plain
JSON at ``<db folder>/settings.json``.
"""

from __future__ import annotations

import json
import logging
import os
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
    use_phase_model: bool = False  # learned release model (phases.use_model), if trained
    accuracy: Literal["auto", "fast", "accurate"] = "auto"


def settings_file(settings: Settings) -> Path:
    """Where the local settings live."""
    return settings.storage.db_file.parent / "settings.json"


def defaults(settings: Settings) -> LocalSettings:
    """Local settings taken from the config file."""
    return LocalSettings(
        coach_model=settings.coach.model,
        pose_backend=settings.pose.backend,
        use_phase_model=settings.phases.use_model,
    )


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
    merged = settings.model_copy(
        update={
            "coach": settings.coach.model_copy(update={"model": local.coach_model}),
            "pose": settings.pose.model_copy(
                update={"backend": local.pose_backend, "accuracy": local.accuracy}
            ),
            "phases": settings.phases.model_copy(update={"use_model": local.use_phase_model}),
        }
    )
    return accuracy_profile(merged)


def accurate_wanted(settings: Settings) -> bool:
    """Whether to use the accurate models: forced, or "auto" with a usable CUDA GPU."""
    mode = os.environ.get("BOWLSENSE_ACCURACY", settings.pose.accuracy)  # tests pin "fast"
    if mode != "auto":
        return mode == "accurate"
    from bowlsense.device import gpu_status

    gpu = gpu_status()
    return bool(gpu.get("enabled") and gpu.get("onnxruntime_cuda") and gpu.get("torch_cuda"))


def accuracy_profile(settings: Settings) -> Settings:
    """Swap in the high-accuracy models (RTMPose-x 384x288 with flip test, YOLO11x with a
    larger ball pass, MotionBERT 3D lifting) when wanted and their files exist; otherwise
    keep the fast ones, so a laptop without them still works."""
    if not accurate_wanted(settings):
        return settings
    models = settings.storage.models_path
    pose, det, ball = settings.pose, settings.detect, settings.ball
    pose_up: dict[str, object] = {"use_flip_test": pose.flip_test}
    if (models / pose.accurate_rtm_model).is_file():
        pose_up |= {"rtm_model": pose.accurate_rtm_model,
                    "rtm_input_size": pose.accurate_rtm_input_size}  # fmt: skip
    if pose.accurate_lift_3d and (models / pose.lift_model).is_file():
        pose_up["lift_3d"] = True
    phases = settings.phases
    if (models / phases.accurate_model_file).is_file():  # release model trained on these poses
        phases = phases.model_copy(update={"model_file": phases.accurate_model_file})
    det_up = {}
    if (models / det.accurate_yolo_model).is_file():
        det_up["yolo_model"] = det.accurate_yolo_model
    return settings.model_copy(
        update={
            "pose": pose.model_copy(update=pose_up),
            "detect": det.model_copy(update=det_up),
            "ball": ball.model_copy(update={"imgsz": ball.accurate_imgsz}),
            "phases": phases,
        }
    )
