"""Optical-flow refinement of the bowling wrist around release.

Motion blur at release is where pose models struggle most: the wrist drops out, loses
confidence or jumps. Around each release the wrist is also tracked with pyramidal
Lucas-Kanade optical flow (forward and backward, with a forward-backward consistency check),
and pose points that are missing, low-confidence or far off that track are replaced by it."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from bowlsense.config import PoseSettings
from bowlsense.geometry.body import joint
from bowlsense.io.video import iter_frames
from bowlsense.schemas import FramePose, Keypoint, PoseSequence

log = logging.getLogger(__name__)
Pt = tuple[float, float]
LK = {"winSize": (21, 21), "maxLevel": 3,
      "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}  # fmt: skip


def _flow(prev: np.ndarray, nxt: np.ndarray, p: Pt, fb_px: float) -> Pt | None:
    """Track one point from ``prev`` to ``nxt``; None if the forward-backward error is big."""
    a = np.array([[p]], dtype=np.float32)
    b, ok, _ = cv2.calcOpticalFlowPyrLK(prev, nxt, a, None, **LK)  # type: ignore[call-overload]
    if b is None or not ok[0][0]:
        return None
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(nxt, prev, b, None, **LK)  # type: ignore[call-overload]
    if back is None or not ok2[0][0] or float(np.hypot(*(back[0][0] - a[0][0]))) > fb_px:
        return None
    return float(b[0][0][0]), float(b[0][0][1])


def _height_px(seq: PoseSequence) -> float:
    hs = [abs(f.keypoints["l_ankle"].y - f.keypoints["nose"].y)  # type: ignore[union-attr]
          for f in seq.frames
          if f.keypoints.get("nose") and f.keypoints.get("l_ankle")]  # fmt: skip
    return float(np.percentile(hs, 80)) if hs else float(seq.height) * 0.6


def _pass(
    order: list[int],
    grays: dict[int, np.ndarray],
    pose: dict[int, Keypoint | None],
    conf: float,
    jump: float,
    fb_px: float,
) -> dict[int, Pt]:
    """One tracking pass in ``order``: returns the flow points used where the pose was bad."""
    out: dict[int, Pt] = {}
    cur: Pt | None = None
    for k, idx in enumerate(order):
        p = pose.get(idx)
        good = p is not None and p.conf >= conf
        if good and (cur is None or np.hypot(p.x - cur[0], p.y - cur[1]) <= jump):  # type: ignore[union-attr]
            cur = (p.x, p.y)  # type: ignore[union-attr]
        elif cur is not None:
            out[idx] = cur  # pose missing, unsure or jumped: trust the flow track
        if cur is not None and k + 1 < len(order):
            cur = _flow(grays[idx], grays[order[k + 1]], cur, fb_px)
    return out


def refine_wrist(
    video: Path, seq: PoseSequence, releases: Sequence[int], settings: PoseSettings
) -> PoseSequence:
    """``seq`` with the bowling wrist refined by optical flow around each release frame."""
    if not seq.frames or seq.fps <= 0:
        return seq
    name = joint(seq.bowling_arm, "bowling", "wrist")
    n = max(2, round(settings.refine_window_s * seq.fps))
    jump = settings.refine_max_jump_h * _height_px(seq)
    by = {f.frame_idx: i for i, f in enumerate(seq.frames)}
    frames = list(seq.frames)
    changed = 0
    for rel in releases:
        lo, hi = max(0, rel - n), rel + n + 1
        grays = {i: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for i, f in iter_frames(video, lo, hi)}
        order = sorted(i for i in grays if i in by)
        if len(order) < 3:
            continue
        pose = {i: frames[by[i]].keypoints.get(name) for i in order}
        fwd = _pass(order, grays, pose, settings.refine_conf, jump, 2.0)
        bwd = _pass(order[::-1], grays, pose, settings.refine_conf, jump, 2.0)
        for i in order:
            pts = [p for p in (fwd.get(i), bwd.get(i)) if p is not None]
            if not pts:
                continue
            x, y = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
            f: FramePose = frames[by[i]]
            kps = dict(f.keypoints)
            kps[name] = Keypoint(x=x, y=y, z=None, conf=settings.refine_conf)
            frames[by[i]] = f.model_copy(update={"keypoints": kps})
            changed += 1
    if changed:
        log.info("wrist refinement: %d frame(s) corrected by optical flow", changed)
    return seq.model_copy(update={"frames": frames})
