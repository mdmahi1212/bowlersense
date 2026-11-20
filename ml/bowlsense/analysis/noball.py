"""Front-foot no-ball screening without pitch calibration: find the white popping-crease line
near the front foot at front-foot contact and check whether the heel landed beyond it.

From a side-on camera the crease runs towards the camera, so it shows as a steep white line at
the bowler's feet. This is a best-effort screen with a clear "not found" outcome."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from bowlsense.config import DeliverySettings
from bowlsense.geometry.body import joint
from bowlsense.io.video import Frame
from bowlsense.schemas import FramePose, NoBallResult


def detect_no_ball(
    frame: Frame,
    pose: FramePose | None,
    arm: Literal["right", "left"],
    height_px: float,
    settings: DeliverySettings,
) -> NoBallResult:
    """Screen the front foot at FFC against an auto-detected popping crease."""
    if pose is None or height_px <= 0:
        return NoBallResult(status="not_found", note="No skeleton at front-foot contact.")
    heel = pose.get(joint(arm, "front", "heel")) or pose.get(joint(arm, "front", "ankle"))
    back = pose.get(joint(arm, "back", "ankle"))
    if heel is None or back is None:
        return NoBallResult(
            status="not_found", note="The feet aren't visible at front-foot contact."
        )
    direction = 1.0 if heel.x >= back.x else -1.0  # the way the bowler is running in the image
    h, w = frame.shape[:2]
    x0, x1 = int(max(0, heel.x - 0.5 * height_px)), int(min(w, heel.x + 0.5 * height_px))
    y0, y1 = int(max(0, heel.y - 0.15 * height_px)), int(min(h, heel.y + 0.08 * height_px))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return NoBallResult(status="not_found", note="The feet are at the edge of the frame.")
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, (0, 0, settings.crease_white_v_min), (180, settings.crease_white_s_max, 255)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    min_len = max(6, int(settings.crease_min_length_h * height_px))
    lines = cv2.HoughLinesP(
        mask, 1, np.pi / 180, threshold=max(8, min_len // 2), minLineLength=min_len, maxLineGap=4
    )
    best: tuple[float, tuple[float, float, float, float]] | None = None
    segs = np.zeros((0, 4)) if lines is None else np.asarray(lines).reshape(-1, 4)
    for x_a, y_a, x_b, y_b in segs:
        ang = math.degrees(math.atan2(abs(y_b - y_a), abs(x_b - x_a) + 1e-9))
        if ang < settings.crease_min_angle_deg:
            continue
        ax, ay, bx, by = x_a + x0, y_a + y0, x_b + x0, y_b + y0
        x_at = ax + (bx - ax) * ((heel.y - ay) / (by - ay)) if by != ay else (ax + bx) / 2
        dist = abs(x_at - heel.x)
        if best is None or dist < best[0]:
            best = (dist, (float(ax), float(ay), float(bx), float(by)))
    if best is None:
        return NoBallResult(
            status="not_found",
            note="No crease line found near the front foot (it may be off camera or worn).",
        )
    ax, ay, bx, by = best[1]
    x_at = ax + (bx - ax) * ((heel.y - ay) / (by - ay)) if by != ay else (ax + bx) / 2
    margin_cm = (heel.x - x_at) * direction / height_px * settings.assumed_height_m * 100.0
    return NoBallResult(
        status="no_ball" if margin_cm > 0 else "legal",
        margin_cm=round(margin_cm, 1),
        line=[(ax, ay), (bx, by)],
        note=(
            "Auto-detected crease line (2D screening): check the front-foot-contact frame. "
            "Distances assume a 1.75 m bowler."
        ),
    )


def rough_speed_kmh(
    points: list[tuple[int, float, float, float]],
    release_frame: int,
    fps: float,
    height_px: float,
    view: str,
    settings: DeliverySettings,
) -> float | None:
    """Ball speed without pitch calibration: the ball's image speed just after release,
    scaled by the bowler's body height (assumed ``assumed_height_m``). Side-on views only,
    since the ball must travel across the image; None when it can't be estimated."""
    if view not in ("side", "unknown") or fps <= 0 or height_px <= 0:
        return None
    last = release_frame + max(2, round(settings.rough_speed_window_s * fps))
    pts = [(f, x, y) for f, x, y, _ in points if release_frame < f <= last]
    if len(pts) < 3:
        return None
    t = np.array([p[0] for p in pts], dtype=float) / fps
    vx = float(np.polyfit(t, [p[1] for p in pts], 1)[0])
    vy = float(np.polyfit(t, [p[2] for p in pts], 1)[0])
    kmh = math.hypot(vx, vy) / height_px * settings.assumed_height_m * 3.6
    lo, hi = settings.rough_speed_range_kmh
    return round(kmh, 1) if lo <= kmh <= hi else None
