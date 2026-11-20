"""Ball detection (YOLO "sports ball") and Kalman tracking after release.

State ``[x, y, vx, vy, ay]`` in pixels: constant velocity plus a constant vertical
acceleration (gravity, in pixels/s^2, estimated as part of the state). Detections are
associated to the prediction within ``gate_px``; up to ``max_missed`` frames without a match
are bridged with predictions (stored with confidence 0). Detection runs on a square window
around the predicted ball, starting at the bowling hand at release, so the small ball keeps
its resolution.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.config import BallSettings, DetectSettings
from bowlsense.detect.person import _device, model_path
from bowlsense.geometry.angles import FloatArray
from bowlsense.io.video import Frame
from bowlsense.schemas import BallTrack

Candidate = tuple[float, float, float]  # x, y, confidence (full-frame pixels)
Detector = Callable[[Frame, tuple[float, float]], list[Candidate]]


class BallKalman:
    """Constant-velocity + constant-vertical-acceleration Kalman filter (pixels)."""

    def __init__(self, x: float, y: float, fps: float, s: BallSettings) -> None:
        """Start at a detection with unknown velocity."""
        dt = 1.0 / fps
        self.F = np.array(
            [
                [1, 0, dt, 0, 0],
                [0, 1, 0, dt, 0.5 * dt * dt],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, dt],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.H = np.zeros((2, 5))
        self.H[0, 0] = self.H[1, 1] = 1.0
        self.R = np.eye(2) * s.meas_noise_px**2
        q = s.accel_noise_px_s2
        g = np.array([0.5 * dt * dt, 0.5 * dt * dt, dt, dt, 1.0])
        self.Q = np.diag(g**2) * np.array([q, q, q, q, q * 0.25]) ** 2 * 1e-6 + np.eye(5) * 1e-3
        self.x = np.array([x, y, 0.0, 0.0, 0.0])
        self.P = np.diag([s.meas_noise_px**2] * 2 + [3000.0**2] * 2 + [5000.0**2])

    def predict(self) -> FloatArray:
        """Advance one frame; returns the predicted position."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return np.asarray(self.x[:2], dtype=np.float64)

    def update(self, z: FloatArray) -> None:
        """Correct with a measured position."""
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(5) - k @ self.H) @ self.P


def track_ball(
    frames: Iterable[tuple[int, Frame]],
    detector: Detector,
    release: int,
    hand: tuple[float, float],
    fps: float,
    settings: BallSettings,
    first_within: int,
) -> BallTrack:
    """Follow the ball from release. ``frames`` must start at (or just before) ``release``.

    The first detection must come within ``first_within`` frames of release and within
    ``gate_px`` of the bowling hand. Returns an empty track when the ball is never found.
    """
    kf: BallKalman | None = None
    points: list[tuple[int, float, float, float]] = []
    missed = 0
    centre = hand
    last = release + round(settings.track_after_release_s * fps)
    for idx, frame in frames:
        if idx < release:
            continue
        if idx > last:
            break
        if kf is None:
            if idx > release + first_within:
                break
            cands = [c for c in detector(frame, centre) if _dist(c, hand) <= settings.gate_px]
            if cands:
                best = max(cands, key=lambda c: c[2])
                kf = BallKalman(best[0], best[1], fps, settings)
                points.append((idx, best[0], best[1], best[2]))
            continue
        pred = kf.predict()
        cands = [c for c in detector(frame, (float(pred[0]), float(pred[1])))
                 if _dist(c, (pred[0], pred[1])) <= settings.gate_px]  # fmt: skip
        if cands:
            best = min(cands, key=lambda c: _dist(c, (pred[0], pred[1])))
            kf.update(np.array(best[:2]))
            points.append((idx, best[0], best[1], best[2]))
            missed = 0
        else:
            missed += 1
            if missed > settings.max_missed:
                break
            points.append((idx, float(pred[0]), float(pred[1]), 0.0))
    while points and points[-1][3] == 0.0:  # don't end on bridged guesses
        points.pop()
    track = BallTrack(points=points, release_frame=points[0][0] if points else None)
    return track.model_copy(update={"bounce_frame": find_bounce(track)})


def _dist(c: tuple[float, ...], p: tuple[float, ...]) -> float:
    return float(np.hypot(c[0] - p[0], c[1] - p[1]))


def find_bounce(track: BallTrack, half_window: int = 4) -> int | None:
    """The bounce: the strongest reversal from moving down the image to moving up.

    For each detected frame, straight lines are fitted to image y over the ``half_window``
    detections before it and after it (inclusive); the bounce is where the slope flips from
    downward to upward by the largest amount. Fitting several frames on each side keeps
    keypoint-sized noise from creating fake reversals.
    """
    pts = [(f, y) for f, _, y, c in track.points if c > 0]
    if len(pts) < 2 * half_window + 1:
        return None
    frames = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    best, best_flip = None, 0.0
    for k in range(half_window, len(pts) - half_window):
        before = np.polyfit(frames[k - half_window : k + 1], ys[k - half_window : k + 1], 1)[0]
        after = np.polyfit(frames[k : k + half_window + 1], ys[k : k + half_window + 1], 1)[0]
        flip = before - after
        if before > 0 and after < 0 and flip > best_flip:
            best, best_flip = int(frames[k]), flip
    return best


class YoloBallDetector:
    """YOLO "sports ball" detections in a square window (upsampled for small balls)."""

    def __init__(self, models_dir: Path, detect: DetectSettings, ball: BallSettings) -> None:
        """Load the YOLO model once."""
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"sync": False})
        self._model: Any = YOLO(str(model_path(models_dir, detect.yolo_model)))
        self.ball = ball
        self._device = _device()

    def __call__(self, frame: Frame, centre: tuple[float, float]) -> list[Candidate]:
        """Ball candidates near ``centre`` in full-frame pixels."""
        h, w = frame.shape[:2]
        half = self.ball.roi_px // 2
        x0 = int(min(max(0, centre[0] - half), max(0, w - self.ball.roi_px)))
        y0 = int(min(max(0, centre[1] - half), max(0, h - self.ball.roi_px)))
        crop = frame[y0 : y0 + self.ball.roi_px, x0 : x0 + self.ball.roi_px]
        res = self._model.predict(crop, classes=[self.ball.yolo_class], conf=self.ball.det_conf,
                                  imgsz=self.ball.imgsz, device=self._device,
                                  verbose=False)  # fmt: skip
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        out = []
        for (a, b, c, d), conf in zip(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), strict=True
        ):
            out.append((x0 + (a + c) / 2.0, y0 + (b + d) / 2.0, float(conf)))
        return out
