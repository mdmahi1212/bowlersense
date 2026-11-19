"""Recording-quality checks and camera-view estimation (spec Section 5)."""

from __future__ import annotations

import numpy as np

from bowlsense.config import VideoSettings
from bowlsense.errors import LowFpsError
from bowlsense.geometry.angles import keypoint_array, keypoint_track, midpoint
from bowlsense.io.video import VideoInfo
from bowlsense.schemas import KEYPOINT_NAMES, PoseSequence, View


def check_video(info: VideoInfo, settings: VideoSettings, allow_low_fps: bool = False) -> list[str]:
    """Frame-rate and resolution checks on the file itself.

    Raises:
        LowFpsError: below ``min_fps_hard`` unless ``allow_low_fps``.
    """
    warnings: list[str] = []
    if info.fps < settings.min_fps_hard:
        if not allow_low_fps:
            raise LowFpsError(info.fps, settings.min_fps_hard)
        warnings.append(
            f"Video is {info.fps:.0f} fps, below the {settings.min_fps_hard:.0f} fps minimum; "
            "phase timing, elbow and release metrics are unreliable."
        )
    elif info.fps < settings.min_fps_recommended:
        warnings.append(
            f"Video is {info.fps:.0f} fps (below {settings.min_fps_recommended:.0f}): release "
            "metrics less reliable. Use slow-motion recording if your phone has it."
        )
    if min(info.width, info.height) < settings.min_height_px:
        warnings.append(
            f"Resolution {info.width}x{info.height} is below {settings.min_height_px}p; "
            "keypoints will be less precise."
        )
    return warnings


def check_pose(seq: PoseSequence, bowler_frames: int, settings: VideoSettings) -> list[str]:
    """Checks on the tracked bowler: visibility and keypoint confidence."""
    warnings: list[str] = []
    n = len(seq.frames)
    visible = bowler_frames / n if n else 0.0
    if visible < settings.min_bowler_visible_frac:
        warnings.append(
            f"The bowler was found in only {visible:.0%} of frames (want "
            f"{settings.min_bowler_visible_frac:.0%}). Keep the whole run-up in frame."
        )
    _, conf = keypoint_array(seq)
    core = [KEYPOINT_NAMES.index(k) for k in KEYPOINT_NAMES[:17]]  # body joints, not feet
    detected = conf[:, core][conf[:, core] > 0]
    mean_conf = float(detected.mean()) if detected.size else 0.0
    if mean_conf < settings.min_mean_keypoint_conf:
        warnings.append(
            f"Mean keypoint confidence is {mean_conf:.2f} (want "
            f"{settings.min_mean_keypoint_conf:.2f}). Improve lighting or move the camera closer."
        )
    return warnings


def estimate_view(seq: PoseSequence, runup: tuple[int, int], settings: VideoSettings) -> View:
    """Side vs front vs behind from the run-up.

    Running across a side-on camera, the shoulders are seen edge-on (narrow); running
    towards or away from the camera they're seen at full width. Front vs behind: a bowler
    approaching the camera grows, one running away shrinks. When size barely changes
    (telephoto lens), left/right shoulder order decides: facing the camera, a person's
    right shoulder is on the image left.
    """
    start, stop = runup
    if stop - start < 2:
        return "unknown"
    l_sh = keypoint_track(seq, "l_shoulder")[start:stop]
    r_sh = keypoint_track(seq, "r_shoulder")[start:stop]
    nose = keypoint_track(seq, "nose")[start:stop, 1]
    ankles = np.fmax(
        keypoint_track(seq, "l_ankle")[start:stop, 1], keypoint_track(seq, "r_ankle")[start:stop, 1]
    )
    size = ankles - nose
    ok = np.isfinite(size) & (size > 0) & np.isfinite(l_sh[:, 0]) & np.isfinite(r_sh[:, 0])
    if ok.sum() < 2:
        return "unknown"
    width = np.abs(l_sh[ok, 0] - r_sh[ok, 0]) / size[ok]
    if float(np.median(width)) < settings.front_view_width_ratio:
        return "side"
    sizes = size[ok]
    k = max(1, len(sizes) // 4)
    change = float(np.median(sizes[-k:]) / np.median(sizes[:k]) - 1.0)
    if change > settings.approach_size_change:
        return "front"
    if change < -settings.approach_size_change:
        return "behind"
    order = np.median(r_sh[ok, 0] - l_sh[ok, 0])
    return "front" if order < 0 else "behind"


def bowler_centre_track(seq: PoseSequence) -> np.ndarray:
    """Hip midpoint per frame (used by tests and diagnostics)."""
    return midpoint(keypoint_track(seq, "l_hip"), keypoint_track(seq, "r_hip"))
