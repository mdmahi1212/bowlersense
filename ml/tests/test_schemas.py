"""Tests for the pydantic data contracts and JSON Schema export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bowlsense import schemas
from bowlsense.schemas import (
    KEYPOINT_NAMES,
    AngleSeries,
    FramePose,
    Keypoint,
    LegalityResult,
    PhaseEvents,
    PoseSequence,
)


def _frame(i: int, **kps: Keypoint | None) -> FramePose:
    return FramePose(frame_idx=i, t=i / 120, keypoints=kps)


def test_keypoint_names_unique_and_complete() -> None:
    assert len(KEYPOINT_NAMES) == 21
    assert len(set(KEYPOINT_NAMES)) == 21
    assert {"l_heel", "r_toe", "nose"} <= set(KEYPOINT_NAMES)


def test_keypoint_conf_bounds() -> None:
    with pytest.raises(ValidationError):
        Keypoint(x=0, y=0, conf=1.5)


def test_frame_pose_rejects_unknown_names() -> None:
    with pytest.raises(ValidationError, match="unknown keypoint"):
        FramePose(frame_idx=0, t=0, keypoints={"left_wrist": Keypoint(x=1, y=1, conf=1)})


def test_frame_pose_allows_missing_joint() -> None:
    f = _frame(0, l_heel=None, nose=Keypoint(x=1, y=2, conf=0.9))
    assert f.get("l_heel") is None
    assert f.get("r_toe") is None
    nose = f.get("nose")
    assert nose is not None and nose.x == 1


def test_pose_sequence_frame_order() -> None:
    PoseSequence(fps=120, width=1920, height=1080, frames=[_frame(0), _frame(1)])
    with pytest.raises(ValidationError, match="increasing"):
        PoseSequence(fps=120, width=1920, height=1080, frames=[_frame(1), _frame(1)])


def test_pose_sequence_rejects_zero_fps() -> None:
    with pytest.raises(ValidationError):
        PoseSequence(fps=0, width=10, height=10, frames=[])


def test_phase_ordering_violations() -> None:
    good = PhaseEvents(
        run_up_start=0,
        bound=10,
        bfc=20,
        ffc=30,
        arm_horizontal=35,
        release=40,
        follow_through_end=60,
        confidence=0.9,
    )
    assert good.ordering_violations() == []
    bad = good.model_copy(update={"ffc": 19})
    assert bad.ordering_violations() == ["expected bfc (20) < ffc (19)"]
    worse = good.model_copy(update={"release": 70})
    assert worse.ordering_violations() == ["expected release (70) < follow_through_end (60)"]
    assert "ORDER" not in PhaseEvents.model_fields


def test_angle_series_lengths_match() -> None:
    AngleSeries(name="elbow", values=[1.0, None], frames=[0, 1])
    with pytest.raises(ValidationError):
        AngleSeries(name="elbow", values=[1.0], frames=[0, 1])


def test_legality_default_note() -> None:
    r = LegalityResult(elbow_extension_deg=10, threshold_deg=15, flagged=False, confidence="low")
    assert "2D video screening only" in r.note
    assert "3D motion capture" in r.note


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Keypoint(x=0, y=0, conf=0.5, w=1)  # type: ignore[call-arg]


def test_json_round_trip() -> None:
    seq = PoseSequence(
        fps=240,
        width=1280,
        height=720,
        view="side",
        bowling_arm="left",
        frames=[_frame(0, nose=Keypoint(x=1, y=2, z=None, conf=0.8))],
    )
    again = PoseSequence.model_validate_json(seq.model_dump_json())
    assert again == seq


def test_json_schema_has_every_model() -> None:
    doc = schemas.json_schema()
    for model in schemas.EXPORTED_MODELS:
        assert model.__name__ in doc["$defs"]
    assert "$ref" in json.dumps(doc["$defs"]["SessionResult"])


def test_export_json_schema(tmp_path: Path) -> None:
    out = schemas.export_json_schema(tmp_path / "nested" / "schema.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["title"] == "BowlSense"


def test_schema_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "s.json"
    schemas.main(["--out", str(target)])
    assert target.is_file()
    assert str(target) in capsys.readouterr().out
