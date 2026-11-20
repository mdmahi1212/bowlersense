"""Data contracts shared by every BowlSense module (pydantic v2).

Run ``python -m bowlsense.schemas --out schema/bowlsense.schema.json`` to export JSON Schema;
the web app generates its TypeScript types from that file.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
    "l_heel",
    "r_heel",
    "l_toe",
    "r_toe",
)

SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("nose", "l_eye"),
    ("nose", "r_eye"),
    ("l_eye", "l_ear"),
    ("r_eye", "r_ear"),
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"),
    ("l_hip", "r_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
    ("l_ankle", "l_heel"),
    ("l_ankle", "l_toe"),
    ("l_heel", "l_toe"),
    ("r_ankle", "r_heel"),
    ("r_ankle", "r_toe"),
    ("r_heel", "r_toe"),
)
"""Bones drawn by renderers (pairs of keypoint names)."""

View = Literal["side", "front", "behind", "unknown"]
Arm = Literal["right", "left"]
RiskBand = Literal["low", "medium", "high"]
Confidence = Literal["low", "medium", "high"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Keypoint(_Model):
    """One detected joint in full-frame pixel coordinates."""

    x: float
    y: float
    z: float | None = None
    conf: float = Field(ge=0, le=1)


class FramePose(_Model):
    """All keypoints of the bowler in one frame. A missing joint maps to ``None``."""

    frame_idx: int = Field(ge=0)
    t: float = Field(ge=0, description="Seconds from the start of the video")
    keypoints: dict[str, Keypoint | None]

    @field_validator("keypoints")
    @classmethod
    def _known_names(cls, v: dict[str, Keypoint | None]) -> dict[str, Keypoint | None]:
        unknown = set(v) - set(KEYPOINT_NAMES)
        if unknown:
            raise ValueError(f"unknown keypoint names: {sorted(unknown)}")
        return v

    def get(self, name: str) -> Keypoint | None:
        """Return the keypoint by name, or ``None`` if missing."""
        return self.keypoints.get(name)


class PoseSequence(_Model):
    """A bowler's pose over time."""

    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    view: View = "unknown"
    bowling_arm: Arm = "right"
    frames: list[FramePose]

    @model_validator(mode="after")
    def _frames_increasing(self) -> PoseSequence:
        idx = [f.frame_idx for f in self.frames]
        if any(b <= a for a, b in pairwise(idx)):
            raise ValueError("frames must be sorted by strictly increasing frame_idx")
        return self


class PhaseEvents(_Model):
    """Frame indices of the key events of one delivery."""

    run_up_start: int = Field(ge=0)
    bound: int = Field(ge=0)
    bfc: int = Field(ge=0)
    ffc: int = Field(ge=0)
    arm_horizontal: int = Field(ge=0)
    release: int = Field(ge=0)
    follow_through_end: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)

    ORDER: ClassVar[tuple[str, ...]] = (
        "run_up_start",
        "bound",
        "bfc",
        "ffc",
        "arm_horizontal",
        "release",
        "follow_through_end",
    )

    def ordering_violations(self) -> list[str]:
        """Return each adjacent pair that breaks the expected strict ordering."""
        out: list[str] = []
        for a, b in pairwise(self.ORDER):
            if not getattr(self, a) < getattr(self, b):
                out.append(f"expected {a} ({getattr(self, a)}) < {b} ({getattr(self, b)})")
        return out


class BallTrack(_Model):
    """Ball positions over time: (frame, x, y, conf)."""

    points: list[tuple[int, float, float, float]]
    release_frame: int | None = None
    bounce_frame: int | None = None


class AngleSeries(_Model):
    """A named angle over time."""

    name: str
    unit: Literal["deg"] = "deg"
    values: list[float | None]
    frames: list[int]

    @model_validator(mode="after")
    def _same_length(self) -> AngleSeries:
        if len(self.values) != len(self.frames):
            raise ValueError("values and frames must have the same length")
        return self


class Fault(_Model):
    """A detected technique fault."""

    code: str
    name: str
    severity: float = Field(ge=0, le=100)
    value: float
    threshold: float
    frames: list[int]
    explanation: str


class FaultReport(_Model):
    """Module C: technique faults for one delivery.

    ``measurements`` holds the value behind every fault that could be measured from this
    camera view (fired or not); ``not_measured`` maps the other codes to the reason.
    """

    faults: list[Fault]
    measurements: dict[str, float]
    not_measured: dict[str, str]
    technique_score: float = Field(ge=0, le=100)


class RiskComponent(_Model):
    """One component of the injury-risk model."""

    name: str
    score: float = Field(ge=0, le=100)
    weight: float = Field(ge=0, le=1)
    drivers: list[str]


class InjuryRisk(_Model):
    """Rule-based injury-risk estimate (not medical advice)."""

    overall: float = Field(ge=0, le=100)
    band: RiskBand
    components: list[RiskComponent]
    top_drivers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Rule-based screening estimate, not a trained medical model and not medical advice."
    )


class InjuryFeatures(_Model):
    """Inputs to the injury-risk model. ``None`` = not measurable from this clip/view."""

    counter_rotation_deg: float | None = None
    mixed_action: bool | None = None
    lateral_flexion_deg: float | None = None
    knee_collapse_deg: float | None = None
    stride_pct: float | None = None
    arm_slot_deg: float | None = None
    arm_angular_velocity_deg_s: float | None = None
    trunk_rotation_speed_deg_s: float | None = None
    fatigue_index: float | None = None


class LegalityResult(_Model):
    """Elbow-extension screening result.

    ``verdict`` is the plain answer shown to the user: ``legal`` (elbow straightened no more
    than the threshold), ``not_legal`` (more than the threshold) or ``cannot_judge`` (the clip
    can't support a measurement; see ``not_screened_reason``). ``verdict_text`` says it in words.

    ``screened`` is False when the clip can't support a trustworthy measurement (low fps,
    camera view, phase events out of order, implausible value); then ``flagged`` is always
    False and ``not_screened_reason`` says why. ``elbow_extension_deg`` is still reported.
    """

    elbow_extension_deg: float
    threshold_deg: float
    flagged: bool
    screened: bool = True
    not_screened_reason: str | None = None
    verdict: Literal["legal", "not_legal", "cannot_judge"] = "legal"
    verdict_text: str = ""
    confidence: Confidence
    note: str = "2D video screening only. The official ICC test uses 3D motion capture."
    front_foot_no_ball: bool | None = None


ActionType = Literal["side_on", "semi_open", "front_on", "mixed", "unknown"]


class AlignmentResult(_Model):
    """Module B: shoulder and hip alignment through the delivery stride.

    Angles are yaw about the vertical axis: 0 = side-on (body line pointing down the pitch),
    90 = front-on, negative = turned past side-on. ``None`` means not measurable from this
    camera view (see ``notes``).
    """

    action_type: ActionType
    shoulder_angle_bfc_deg: float | None
    hip_angle_bfc_deg: float | None
    hip_shoulder_diff_bfc_deg: float | None
    counter_rotation_deg: float | None
    counter_rotation_flagged: bool | None
    hip_shoulder_sep_ffc_deg: float | None
    notes: list[str] = Field(default_factory=list)


class BallMetrics(_Model):
    """Module F: speed, length, line and pitch position (``None`` = not measurable, see
    ``notes``)."""

    speed_kmh: float | None
    length: str | None
    line: str | None
    pitch_xy: tuple[float, float] | None
    notes: list[str]


class ReferenceComparison(_Model):
    """How closely an action matches one reference action (module G)."""

    name: str
    similarity_pct: float = Field(ge=0, le=100)
    deviation_deg: dict[str, float]  # mean absolute difference per joint along the DTW path


class ConsistencyResult(_Model):
    """Consistency across a session's deliveries (module G)."""

    score: float | None = Field(default=None, ge=0, le=100)  # None with < 2 deliveries
    variation_pct: dict[str, float]
    n_deliveries: int = Field(ge=0)


class DeliveryTypeResult(_Model):
    """Pace or spin (learned classifier), with the probability of pace."""

    delivery_type: Literal["fast", "spin"]
    p_fast: float = Field(ge=0, le=1)


class DeliveryAnalysis(_Model):
    """Modules A-D (and E's per-delivery inputs) for one delivery."""

    height_px: float
    phases: PhaseEvents
    legality: LegalityResult
    alignment: AlignmentResult
    faults: FaultReport
    injury: InjuryRisk
    fatigue_input: FatigueInput
    delivery_type: DeliveryTypeResult | None = None
    release_source: Literal["rules", "model"] = "rules"
    ball: BallMetrics | None = None
    ball_track: BallTrack | None = None
    comparisons: list[ReferenceComparison] = Field(default_factory=list)


class DeliveryMetrics(_Model):
    """Everything measured for one delivery.

    Phase frames and ``angles`` frames are numbered within the delivery's clip;
    ``clip_start`` is the clip's first frame in the source video. ``None`` = not measurable
    from this camera view.
    """

    delivery_idx: int = Field(ge=0)
    phases: PhaseEvents
    action_type: str
    legality: LegalityResult
    faults: list[Fault]
    technique_score: float = Field(ge=0, le=100)
    injury: InjuryRisk
    speed_kmh: float | None = None
    length: str | None = None
    line: str | None = None
    pitch_xy: tuple[float, float] | None = None
    release_height_pct: float | None = None
    stride_pct: float | None = None
    angles: list[AngleSeries]
    clip_start: int = Field(default=0, ge=0)
    clip_stop: int | None = None
    alignment: AlignmentResult | None = None
    fault_measurements: dict[str, float] = Field(default_factory=dict)
    not_measured: dict[str, str] = Field(default_factory=dict)
    delivery_type: DeliveryTypeResult | None = None
    release_source: Literal["rules", "model"] = "rules"
    ball_notes: list[str] = Field(default_factory=list)
    ball_track: BallTrack | None = None
    comparisons: list[ReferenceComparison] = Field(default_factory=list)
    keyframes: dict[str, str] = Field(default_factory=dict)  # event -> artifact path


class FatigueInput(_Model):
    """Per-delivery numbers the fatigue module tracks. ``None`` = not measured."""

    delivery_idx: int = Field(ge=0)
    runup_speed_h_s: float | None = None
    release_height_pct: float | None = None
    wrist_speed_h_s: float | None = None
    technique_score: float | None = None
    fault_count: int = Field(ge=0)


class FatigueDrift(_Model):
    """Percent change of one delivery's metrics versus the session baseline."""

    delivery_idx: int = Field(ge=0)
    runup_speed_pct: float | None
    release_height_pct: float | None
    wrist_speed_pct: float | None
    technique_score_pct: float | None
    fault_count_delta: int
    fatigue_index: float | None = Field(default=None, ge=0, le=100)


class FatigueResult(_Model):
    """Session-level fatigue trend."""

    baseline_deliveries: int = Field(ge=0)
    drifts: list[FatigueDrift]
    fatigue_index: float = Field(ge=0, le=100)
    alert: bool
    alert_reasons: list[str]
    notes: list[str] = Field(default_factory=list)


class WorkloadResult(_Model):
    """Acute:chronic workload ratio. Ratio fields are ``None`` without enough history."""

    balls_this_session: int = Field(ge=0)
    acute_load: float | None = None
    chronic_load: float | None = None
    acwr: float | None = None
    flagged: bool = False
    note: str = ""


class SessionResult(_Model):
    """The full result of analysing one session (one video).

    ``consistency_score`` is ``None`` with fewer than 2 deliveries. ``artifacts`` maps names
    (overlay, report, pose, keyframe paths) to paths relative to the output location.
    """

    session_id: str
    deliveries: list[DeliveryMetrics]
    consistency_score: float | None = Field(default=None, ge=0, le=100)
    fatigue: FatigueResult
    workload: WorkloadResult
    similar_pros: list[tuple[str, float]]
    coach_report: str
    quality_warnings: list[str]
    artifacts: dict[str, str]
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(ge=0)
    view: View
    bowling_arm: Arm
    pose_backend: str = ""
    bowler_track_id: int | None = None
    video_fps: float = 0.0  # the file's playback rate (``fps`` is the real-time rate)
    slow_motion_factor: float = 1.0
    slow_motion_source: Literal["none", "auto", "user"] = "none"
    consistency: ConsistencyResult | None = None
    coach_source: Literal["ollama", "template"] = "template"
    coach_note: str = ""


class TrackedPerson(_Model):
    """One person the tracker followed: their box (x1, y1, x2, y2) in each frame seen."""

    track_id: int
    frames: list[int]
    boxes: list[tuple[float, float, float, float]]


class PeopleTracks(_Model):
    """Everyone tracked in a session's video, and which one was analysed as the bowler."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    chosen_track_id: int | None
    chosen_by: Literal["auto", "user"]
    people: list[TrackedPerson]


EXPORTED_MODELS: tuple[type[BaseModel], ...] = (
    Keypoint,
    FramePose,
    PoseSequence,
    PhaseEvents,
    BallTrack,
    BallMetrics,
    AngleSeries,
    Fault,
    FaultReport,
    RiskComponent,
    InjuryRisk,
    InjuryFeatures,
    LegalityResult,
    AlignmentResult,
    ReferenceComparison,
    ConsistencyResult,
    DeliveryTypeResult,
    DeliveryAnalysis,
    DeliveryMetrics,
    FatigueInput,
    FatigueDrift,
    FatigueResult,
    WorkloadResult,
    SessionResult,
    TrackedPerson,
    PeopleTracks,
)


def json_schema(
    models: Sequence[type[BaseModel]] = EXPORTED_MODELS,
    mode: Literal["validation", "serialization"] = "validation",
) -> dict[str, Any]:
    """One JSON Schema document with every model under ``$defs`` (and listed in
    ``properties`` so code generators emit them all)."""
    defs: dict[str, Any] = {}
    for model in models:
        schema = model.model_json_schema(ref_template="#/$defs/{model}", mode=mode)
        defs.update(schema.pop("$defs", {}))
        defs[model.__name__] = schema
    if mode == "serialization":  # every field is always present in what the API sends
        for d in defs.values():
            if "properties" in d:
                d["required"] = list(d["properties"])
    names = sorted(m.__name__ for m in models)
    return {
        "title": "BowlSense",
        "type": "object",
        "properties": {n: {"$ref": f"#/$defs/{n}"} for n in names},
        "$defs": dict(sorted(defs.items())),
    }


def export_json_schema(path: Path, *, with_api: bool = False) -> Path:
    """Write the JSON Schema to ``path`` (parent folders are created) and return it.

    ``with_api`` adds the HTTP request/response bodies and uses serialization mode (what the
    API sends), for generating the web app's TypeScript types.
    """
    models: Sequence[type[BaseModel]] = EXPORTED_MODELS
    mode: Literal["validation", "serialization"] = "validation"
    if with_api:
        from bowlsense.api import API_MODELS

        models, mode = (*EXPORTED_MODELS, *API_MODELS), "serialization"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(models, mode), indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: export the JSON Schema."""
    parser = argparse.ArgumentParser(description="Export BowlSense JSON Schema")
    parser.add_argument("--out", type=Path, default=Path("schema/bowlsense.schema.json"))
    parser.add_argument("--with-api", action="store_true", help="include API bodies (web types)")
    args = parser.parse_args(argv)
    print(export_json_schema(args.out, with_api=args.with_api))


if __name__ == "__main__":
    main()
