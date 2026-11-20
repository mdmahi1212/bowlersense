"""Module F: ball speed, length, line and pitch-map position from a ball track.

Speed is measured in the vertical plane along the pitch (see ``geometry.pitch.along_pitch``):
distance along the pitch of each tracked ball position over the first 5-10 frames after
release, fitted against time. It needs a calibration and a side-on camera; otherwise it is
``None`` with a note, never a guess. The bounce point is on the ground, so the ground
homography gives the pitch-map position, then length (from the batter's stumps) and line.
"""

from __future__ import annotations

import numpy as np

from bowlsense.config import DeliverySettings
from bowlsense.geometry.pitch import (
    CREASE_LENGTH_M,
    POPPING_TO_STUMPS_M,
    PitchCalibration,
    along_pitch,
    image_to_pitch,
    is_side_on,
)
from bowlsense.schemas import BallMetrics, BallTrack

BATTER_STUMPS_X = CREASE_LENGTH_M + POPPING_TO_STUMPS_M


def length_zone(distance_m: float, s: DeliverySettings) -> str:
    """Length name from the distance (m) between the bounce and the batter's stumps."""
    z = s.length_zones_m
    if distance_m < z.yorker:
        return "yorker"
    if distance_m < z.full:
        return "full"
    if distance_m < z.good:
        return "good"
    if distance_m < z.short:
        return "short"
    return "bouncer"


def line_band(lateral_m: float, s: DeliverySettings) -> str:
    """Line from the lateral distance (m, + = bowler's left) from middle stump.

    For a right-handed batter the off side is the bowler's left.
    """
    off = lateral_m if s.batter_hand == "right" else -lateral_m
    b = s.line_bands_m
    if abs(off) <= b["middle"]:
        return "middle"
    side = "off" if off > 0 else "leg"
    if abs(off) <= b["stump"]:
        return f"{side} stump"
    if abs(off) <= b["wide"]:
        return f"outside {side}"
    return "wide"


def ground_error_m(
    cal: PitchCalibration, u: float, v: float, error_px: float
) -> tuple[float, float]:
    """How far (m) along and across the pitch a ground point moves for ``error_px`` of image
    error at pixel ``(u, v)`` (worst case over image directions), from the homography's local
    derivatives. Grazing camera angles make one axis very coarse."""
    base = image_to_pitch(cal, np.array([[u, v]]))[0]
    du = image_to_pitch(cal, np.array([[u + 1.0, v]]))[0] - base
    dv = image_to_pitch(cal, np.array([[u, v + 1.0]]))[0] - base
    jac = np.stack([du, dv], axis=1)  # rows: pitch x, pitch y; columns: per pixel u, v
    along = float(np.linalg.norm(jac[0])) * error_px
    across = float(np.linalg.norm(jac[1])) * error_px
    return along, across


def ball_speed_kmh(
    track: BallTrack, cal: PitchCalibration, fps: float, s: DeliverySettings
) -> float | None:
    """Speed over the first ``speed_frames`` detected frames after release (km/h)."""
    lo, hi = s.speed_frames
    pts = [(f, x) for f, x, _, c in track.points if c > 0][:hi]
    if len(pts) < lo:
        return None
    frames = np.array([f for f, _ in pts], dtype=np.float64)
    dist = along_pitch(cal, np.array([x for _, x in pts]))
    ok = np.isfinite(dist)
    if ok.sum() < lo:
        return None
    slope = np.polyfit(frames[ok] / fps, dist[ok], 1)[0]
    return float(abs(slope) * 3.6)


def _empty(note: str) -> BallMetrics:
    return BallMetrics(speed_kmh=None, length=None, line=None, pitch_xy=None, notes=[note])


def assess_ball(
    track: BallTrack, cal: PitchCalibration | None, fps: float, s: DeliverySettings
) -> BallMetrics:
    """Module F for one delivery."""
    notes: list[str] = []
    if not track.points:
        return _empty("the ball was not found after release")
    if cal is None:
        return _empty("needs pitch calibration (click the 4 crease corners)")
    speed: float | None = None
    if is_side_on(cal):
        speed = ball_speed_kmh(track, cal, fps, s)
        if speed is None:
            notes.append(
                f"too few ball detections after release for speed (need {s.speed_frames[0]})"
            )
    else:
        notes.append("speed needs a side-on camera (the ball travels along the camera's view)")
    length = line = None
    pitch_xy: tuple[float, float] | None = None
    if track.bounce_frame is not None:
        bounce = next(p for p in track.points if p[0] == track.bounce_frame)
        px, py = image_to_pitch(cal, np.array([[bounce[1], bounce[2]]]))[0]
        if np.isfinite(px) and np.isfinite(py):
            pitch_xy = (float(px), float(py))
            err_x, err_y = ground_error_m(cal, bounce[1], bounce[2], s.position_error_px)
            if err_x <= s.max_length_error_m:
                length = length_zone(BATTER_STUMPS_X - float(px), s)
            else:
                notes.append(
                    f"length not reported: +-{err_x:.1f} m uncertain from this camera angle "
                    "(a side-on camera measures length best)"
                )
            if err_y <= s.max_line_error_m:
                line = line_band(float(py), s)
            else:
                notes.append(
                    f"line not reported: +-{err_y:.2f} m uncertain from this camera angle "
                    "(a camera behind the bowler measures line best)"
                )
    else:
        notes.append("no bounce found in the ball track")
    return BallMetrics(speed_kmh=speed, length=length, line=line, pitch_xy=pitch_xy, notes=notes)
