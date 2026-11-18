"""Data contracts shared by every BowlSense module (pydantic v2).

Run ``python -m bowlsense.schemas --out schema/bowlsense.schema.json`` to export JSON Schema;
the web app generates its TypeScript types from that file.
"""

from __future__ import annotations

import argparse
import json
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
    disclaimer: str = (
        "Rule-based screening estimate, not a trained medical model and not medical advice."
    )


class LegalityResult(_Model):
    """Elbow-extension screening result."""

    elbow_extension_deg: float
    threshold_deg: float
    flagged: bool
    confidence: Confidence
    note: str = "2D video screening only. The official ICC test uses 3D motion capture."
    front_foot_no_ball: bool | None = None


class DeliveryMetrics(_Model):
    """Everything measured for one delivery."""

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
    release_height_pct: float
    stride_pct: float
    angles: list[AngleSeries]


class FatigueDrift(_Model):
    """Percent change of one delivery's metrics versus the session baseline."""

    delivery_idx: int = Field(ge=0)
    runup_speed_pct: float | None
    release_height_pct: float | None
    wrist_speed_pct: float | None
    technique_score_pct: float | None
    fault_count_delta: int


class FatigueResult(_Model):
    """Session-level fatigue trend."""

    baseline_deliveries: int = Field(ge=0)
    drifts: list[FatigueDrift]
    fatigue_index: float = Field(ge=0, le=100)
    alert: bool
    alert_reasons: list[str]


class WorkloadResult(_Model):
    """Acute:chronic workload ratio. Ratio fields are ``None`` without enough history."""

    balls_this_session: int = Field(ge=0)
    acute_load: float | None = None
    chronic_load: float | None = None
    acwr: float | None = None
    flagged: bool = False
    note: str = ""


class SessionResult(_Model):
    """The full result of analysing one session (one video)."""

    session_id: str
    deliveries: list[DeliveryMetrics]
    consistency_score: float = Field(ge=0, le=100)
    fatigue: FatigueResult
    workload: WorkloadResult
    similar_pros: list[tuple[str, float]]
    coach_report: str
    quality_warnings: list[str]
    artifacts: dict[str, str]


EXPORTED_MODELS: tuple[type[BaseModel], ...] = (
    Keypoint,
    FramePose,
    PoseSequence,
    PhaseEvents,
    BallTrack,
    AngleSeries,
    Fault,
    RiskComponent,
    InjuryRisk,
    LegalityResult,
    DeliveryMetrics,
    FatigueDrift,
    FatigueResult,
    WorkloadResult,
    SessionResult,
)


def json_schema() -> dict[str, Any]:
    """Return one JSON Schema document with every exported model under ``$defs``."""
    defs: dict[str, Any] = {}
    for model in EXPORTED_MODELS:
        schema = model.model_json_schema(ref_template="#/$defs/{model}")
        defs.update(schema.pop("$defs", {}))
        defs[model.__name__] = schema
    return {
        "title": "BowlSense",
        "$defs": dict(sorted(defs.items())),
    }


def export_json_schema(path: Path) -> Path:
    """Write the JSON Schema to ``path`` (parent folders are created) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(), indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: export the JSON Schema."""
    parser = argparse.ArgumentParser(description="Export BowlSense JSON Schema")
    parser.add_argument("--out", type=Path, default=Path("schema/bowlsense.schema.json"))
    args = parser.parse_args(argv)
    print(export_json_schema(args.out))


if __name__ == "__main__":
    main()
