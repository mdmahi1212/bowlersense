"""Image mode (spec 7J): one photo -> pose -> the posture checks that work on a single frame.

A single frame has no timing, so only posture at the shown phase can be checked: front-knee
angle, trunk lean, head position over the front foot and arm slot. The user says which phase
the photo shows (BFC / FFC / release) or lets ``auto`` guess it. Elbow legality, speeds,
sequencing and anything else that needs motion is not available from one image.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field

from bowlsense.config import Settings
from bowlsense.detect.person import BBox
from bowlsense.errors import InvalidInputError, NoBowlerDetectedError, PoseQualityError
from bowlsense.geometry.angles import FloatArray, joint_angle
from bowlsense.geometry.body import joint
from bowlsense.pose.base import PoseBackend, load_backend
from bowlsense.schemas import KEYPOINT_NAMES, Arm, FramePose, Keypoint

log = logging.getLogger(__name__)
Phase = Literal["bfc", "ffc", "release"]
PhaseChoice = Literal["auto", "bfc", "ffc", "release"]
ImageView = Literal["side", "front"]
NOTE = (
    "Single-image screening: posture at one moment only. Elbow legality, speeds and timing "
    "need a video."
)


class ImageCheck(BaseModel):
    """One posture check. ``value`` is ``None`` when the needed joints weren't visible."""

    code: str
    name: str
    value: float | None
    unit: str
    threshold: float | None = None
    flagged: bool | None = None
    note: str


class ImageAnalysis(BaseModel):
    """Result of analysing one image."""

    phase: Phase
    phase_source: Literal["user", "auto"]
    view: ImageView
    view_source: Literal["user", "auto"]
    bowling_arm: Arm
    arm_source: Literal["user", "auto"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    height_px: float | None
    keypoints: dict[str, Keypoint | None]
    checks: list[ImageCheck]
    warnings: list[str]
    annotated: str | None = None  # storage path of the annotated image (API) or file name
    note: str = NOTE


def read_image(path: Path) -> np.ndarray:
    """Load an image as BGR.

    Raises:
        InvalidInputError: if it is missing or not a readable image.
    """
    if not path.is_file():
        raise InvalidInputError(f"Image not found: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)  # handles non-ASCII Windows paths
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise InvalidInputError(f"{path.name} is not a readable JPG or PNG image.")
    return np.asarray(image, dtype=np.uint8)


def detect_person(image: np.ndarray, settings: Settings) -> BBox:
    """The largest person YOLO finds in the image.

    Raises:
        NoBowlerDetectedError: if there is nobody in the image.
    """
    from bowlsense.detect.person import PERSON_CLASS, _device, _new_tracker_model, model_path

    weights = model_path(settings.storage.models_path, settings.detect.yolo_model)
    result = _new_tracker_model(weights).predict(
        image, classes=[PERSON_CLASS], conf=settings.detect.person_conf, device=_device(),
        verbose=False,
    )[0]  # fmt: skip
    boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().numpy().tolist()
    if not boxes:
        raise NoBowlerDetectedError("No person found in the image. Use a photo of the bowler.")
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return (float(x1), float(y1), float(x2), float(y2))


def _point(kp: dict[str, Keypoint | None], name: str) -> FloatArray:
    k = kp.get(name)
    return np.array([np.nan, np.nan]) if k is None else np.array([k.x, k.y], dtype=np.float64)


def _mid(kp: dict[str, Keypoint | None], a: str, b: str) -> FloatArray:
    return np.asarray((_point(kp, a) + _point(kp, b)) / 2.0, dtype=np.float64)


def _finite(*points: FloatArray) -> bool:
    return all(bool(np.isfinite(p).all()) for p in points)


def guess_arm(kp: dict[str, Keypoint | None]) -> Arm:
    """The arm whose wrist is higher above its shoulder (right when unclear)."""
    lift = {}
    for side, arm in (("r", "right"), ("l", "left")):
        w, s = _point(kp, f"{side}_wrist"), _point(kp, f"{side}_shoulder")
        lift[arm] = float(s[1] - w[1]) if _finite(w, s) else -np.inf
    return "left" if lift["left"] > lift["right"] else "right"


def guess_view(kp: dict[str, Keypoint | None], settings: Settings) -> ImageView:
    """Side vs front from the hip width relative to body size (hips turn less than shoulders)."""
    lh, rh = _point(kp, "l_hip"), _point(kp, "r_hip")
    nose = _point(kp, "nose")
    ankles = [_point(kp, "l_ankle"), _point(kp, "r_ankle")]
    low = max((a[1] for a in ankles if _finite(a)), default=np.nan)
    if not _finite(lh, rh, nose) or not np.isfinite(low) or low <= nose[1]:
        return "side"
    ratio = abs(lh[0] - rh[0]) / (low - nose[1])
    return "front" if ratio > settings.image.front_hip_width_ratio else "side"


def guess_phase(kp: dict[str, Keypoint | None], arm: Arm, height_px: float | None) -> Phase:
    """Release if the bowling wrist is above the head; FFC if the feet are a stride apart;
    otherwise BFC."""
    wrist, nose = _point(kp, joint(arm, "bowling", "wrist")), _point(kp, "nose")
    if _finite(wrist, nose) and wrist[1] < nose[1]:
        return "release"
    fa, ba = _point(kp, joint(arm, "front", "ankle")), _point(kp, joint(arm, "back", "ankle"))
    if height_px and _finite(fa, ba) and np.linalg.norm(fa - ba) > 0.45 * height_px:
        return "ffc"
    return "bfc"


def body_height_px(kp: dict[str, Keypoint | None], factor: float) -> float | None:
    """Standing height estimate from nose-to-lowest-ankle (approximate in a bent posture)."""
    nose = _point(kp, "nose")
    ankles = [_point(kp, n) for n in ("l_ankle", "r_ankle")]
    low = max((a[1] for a in ankles if _finite(a)), default=np.nan)
    if not _finite(nose) or not np.isfinite(low) or low <= nose[1]:
        return None
    return float((low - nose[1]) * factor)


def _knee_angle(
    kp: dict[str, Keypoint | None], arm: Arm, role: Literal["front", "back"]
) -> float | None:
    hip, knee, ankle = (_point(kp, joint(arm, role, p)) for p in ("hip", "knee", "ankle"))
    return float(joint_angle(hip, knee, ankle)) if _finite(hip, knee, ankle) else None


def _limit(value: float | None, threshold: float, applies: bool, above: bool = True) -> bool | None:
    """Flag against a threshold only when measured and when the check applies to this phase."""
    if value is None or not applies:
        return None
    return value > threshold if above else value < threshold


def _trunk_lean(kp: dict[str, Keypoint | None]) -> float | None:
    sh, hp = _mid(kp, "l_shoulder", "r_shoulder"), _mid(kp, "l_hip", "r_hip")
    if not _finite(sh, hp) or hp[1] <= sh[1]:
        return None
    return float(np.degrees(np.arctan2(abs(sh[0] - hp[0]), hp[1] - sh[1])))


def _arm_slot(kp: dict[str, Keypoint | None], arm: Arm) -> float | None:
    s, w = _point(kp, joint(arm, "bowling", "shoulder")), _point(kp, joint(arm, "bowling", "wrist"))
    return float(np.degrees(np.arctan2(abs(w[0] - s[0]), s[1] - w[1]))) if _finite(s, w) else None


def posture_checks(
    kp: dict[str, Keypoint | None],
    phase: Phase,
    view: ImageView,
    arm: Arm,
    height_px: float | None,
    settings: Settings,
) -> list[ImageCheck]:
    """The single-frame checks for this phase and view."""
    cfg, faults = settings.image, settings.faults
    at_release = phase == "release"
    checks: list[ImageCheck] = []
    if phase in ("ffc", "release"):
        knee = _knee_angle(kp, arm, "front")
        checks.append(
            ImageCheck(
                code="FRONT_KNEE",
                name="Front-knee brace",
                value=knee,
                unit="deg",
                threshold=cfg.front_knee_min_deg,
                flagged=_limit(knee, cfg.front_knee_min_deg, True, above=False),
                note=f"A braced front leg is about {cfg.front_knee_min_deg:.0f} deg or "
                "straighter from front-foot contact to release.",
            )
        )
    else:
        checks.append(
            ImageCheck(
                code="BACK_KNEE",
                name="Back-knee angle at back-foot contact",
                value=_knee_angle(kp, arm, "back"),
                unit="deg",
                note="Shown for reference; no limit at back-foot contact.",
            )
        )
    lean = _trunk_lean(kp)
    if view == "front":
        checks.append(
            ImageCheck(
                code="LATERAL_FLEXION",
                name="Trunk side-bend",
                value=lean,
                unit="deg",
                threshold=faults.lateral_flexion_deg,
                flagged=_limit(lean, faults.lateral_flexion_deg, at_release),
                note="Side-bend away from the bowling arm loads the lower back (checked at "
                "release).",
            )
        )
        nose, front_ankle = _point(kp, "nose"), _point(kp, joint(arm, "front", "ankle"))
        offset = None
        if height_px and _finite(nose, front_ankle):
            offset = float(abs(nose[0] - front_ankle[0]) / height_px * 100.0)
        checks.append(
            ImageCheck(
                code="HEAD_FALL",
                name="Head over the front foot",
                value=offset,
                unit="% of height",
                threshold=faults.head_fall_pct,
                flagged=_limit(offset, faults.head_fall_pct, at_release),
                note="The head should stay over the front foot at release.",
            )
        )
    else:
        checks.append(
            ImageCheck(
                code="TRUNK_LEAN",
                name="Trunk lean from vertical",
                value=lean,
                unit="deg",
                note="Forward lean in a side view; side-bend needs a front view.",
            )
        )
    if at_release:
        slot = _arm_slot(kp, arm)
        if slot is None:
            label = ""
        elif slot <= cfg.arm_slot_high_deg:
            label = " (high)"
        elif slot >= cfg.arm_slot_round_deg:
            label = " (round-arm)"
        else:
            label = " (three-quarter)"
        checks.append(
            ImageCheck(
                code="ARM_SLOT",
                name=f"Arm slot{label}",
                value=slot,
                unit="deg from vertical",
                note="0 deg = arm straight up. Very round-arm slots load the shoulder more.",
            )
        )
    return checks


def analyze_keypoints(
    kp: dict[str, Keypoint | None],
    width: int,
    height: int,
    settings: Settings,
    *,
    phase: PhaseChoice = "auto",
    view: ImageView | Literal["auto"] = "auto",
    bowling_arm: Arm | None = None,
) -> ImageAnalysis:
    """Posture checks on one frame's keypoints (the core of image mode).

    Raises:
        PoseQualityError: if too few body joints are visible.
    """
    kp = {name: kp.get(name) for name in KEYPOINT_NAMES}
    core = ("l_shoulder", "r_shoulder", "l_hip", "r_hip", "l_knee", "r_knee")
    if sum(kp[n] is not None for n in core) < 4:
        raise PoseQualityError(
            "Too few body joints were found in the image. Use a sharp photo with the whole body "
            "in frame."
        )
    arm = bowling_arm or guess_arm(kp)
    h = body_height_px(kp, settings.body.height_factor)
    chosen_phase = guess_phase(kp, arm, h) if phase == "auto" else phase
    chosen_view = guess_view(kp, settings) if view == "auto" else view
    warnings = []
    if h is None:
        warnings.append("Couldn't estimate body height (head or feet not visible).")
    low = [n for n, k in kp.items() if k is not None and k.conf < settings.pose.min_conf]
    if low:
        warnings.append(f"Low-confidence joints: {', '.join(sorted(low))}.")
    return ImageAnalysis(
        phase=chosen_phase,
        phase_source="user" if phase != "auto" else "auto",
        view=chosen_view,
        view_source="user" if view != "auto" else "auto",
        bowling_arm=arm,
        arm_source="user" if bowling_arm else "auto",
        width=width,
        height=height,
        height_px=h,
        keypoints=kp,
        checks=posture_checks(kp, chosen_phase, chosen_view, arm, h, settings),
        warnings=warnings,
    )


def annotate(image: np.ndarray, result: ImageAnalysis, settings: Settings) -> np.ndarray:
    """The image with the skeleton, the knee angle and a phase banner."""
    from bowlsense.render.overlay import draw_pose

    pose = FramePose(frame_idx=0, t=0.0, keypoints=result.keypoints)
    angles = {}
    knee = next((c for c in result.checks if c.code in ("FRONT_KNEE", "BACK_KNEE")), None)
    if knee is not None and knee.value is not None:
        role: Literal["front", "back"] = "front" if knee.code == "FRONT_KNEE" else "back"
        angles[joint(result.bowling_arm, role, "knee")] = ("Knee", knee.value)
    label = f"Image mode: {result.phase.upper()} ({result.phase_source})"
    return draw_pose(image, pose, settings.render, label=label, angles=angles)


def analyze_image(
    path: Path,
    settings: Settings,
    *,
    phase: PhaseChoice = "auto",
    view: ImageView | Literal["auto"] = "auto",
    bowling_arm: Arm | None = None,
    backend: PoseBackend | None = None,
    annotated_out: Path | None = None,
) -> ImageAnalysis:
    """Photo -> person box -> pose -> posture checks (and an annotated copy if asked)."""
    image = read_image(path)
    h, w = image.shape[:2]
    box = detect_person(image, settings)
    pose_backend = backend or load_backend(
        settings.pose.backend, settings.pose, settings.storage.models_path
    )
    kp = pose_backend.estimate(image, box)
    result = analyze_keypoints(kp, w, h, settings, phase=phase, view=view, bowling_arm=bowling_arm)
    if annotated_out is not None:
        annotated_out.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", annotate(image, result, settings))
        if not ok:
            raise InvalidInputError("Couldn't write the annotated image.")
        annotated_out.write_bytes(buf.tobytes())
        result = result.model_copy(update={"annotated": annotated_out.name})
    return result
