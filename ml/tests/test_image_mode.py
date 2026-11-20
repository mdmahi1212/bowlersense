"""Phase 15: image mode (single-frame posture checks)."""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pytest

from bowlsense.config import Settings, load_settings
from bowlsense.errors import InvalidInputError, NoBowlerDetectedError, PoseQualityError
from bowlsense.image_mode import (
    ImageAnalysis,
    ImageCheck,
    analyze_image,
    analyze_keypoints,
    guess_arm,
    guess_phase,
    read_image,
)
from bowlsense.schemas import Keypoint
from synth.skeleton_gen import SynthParams, SynthResult, generate

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)


@cache
def _settings() -> Settings:
    return load_settings()


@cache
def _synth(**kw: object) -> SynthResult:
    return generate(SynthParams.model_validate({"height_px": 400.0, **kw}))


def _frame(r: SynthResult, event: str) -> dict[str, Keypoint | None]:
    return dict(r.sequence.frames[int(getattr(r.phases, event))].keypoints)


def _check(result: ImageAnalysis, code: str) -> ImageCheck:
    return next(c for c in result.checks if c.code == code)


@pytest.mark.parametrize("arm", ["right", "left"])
def test_guesses_arm_and_phase_on_synthetic_frames(arm: str) -> None:
    r = _synth(bowling_arm=arm)
    h = 400.0 * 1.0
    rel = _frame(r, "release")
    assert guess_arm(rel) == arm
    assert guess_phase(rel, arm, h) == "release"
    assert guess_phase(_frame(r, "ffc"), arm, h) == "ffc"


def test_front_knee_brace_flags_collapse() -> None:
    s = _settings()
    firm = analyze_keypoints(_frame(_synth(), "release"), 1280, 720, s, phase="release")
    soft = analyze_keypoints(
        _frame(_synth(knee_flex_change_deg=35.0), "release"), 1280, 720, s, phase="release"
    )
    k_firm, k_soft = _check(firm, "FRONT_KNEE"), _check(soft, "FRONT_KNEE")
    assert k_firm.flagged is False and k_soft.flagged is True
    assert k_firm.value is not None and k_soft.value is not None
    assert k_firm.value - k_soft.value > 20
    assert firm.phase_source == "user" and firm.bowling_arm == "right"


def test_release_checks_side_and_front_views() -> None:
    s = _settings()
    side = analyze_keypoints(_frame(_synth(), "release"), 1280, 720, s, phase="auto")
    assert side.phase == "release" and side.phase_source == "auto" and side.view == "side"
    codes = {c.code for c in side.checks}
    assert {"FRONT_KNEE", "TRUNK_LEAN", "ARM_SLOT"} <= codes and "HEAD_FALL" not in codes
    slot = _check(side, "ARM_SLOT")
    assert slot.value is not None and slot.value < 45
    front = analyze_keypoints(
        _frame(_synth(view="front", lateral_flexion_deg=40.0), "release"), 1280, 720, s,
        phase="release", view="front",
    )  # fmt: skip
    lateral = _check(front, "LATERAL_FLEXION")
    assert lateral.flagged is True
    assert {"HEAD_FALL", "LATERAL_FLEXION"} <= {c.code for c in front.checks}


def test_bfc_reports_back_knee_only() -> None:
    result = analyze_keypoints(_frame(_synth(), "bfc"), 1280, 720, _settings(), phase="bfc")
    codes = [c.code for c in result.checks]
    assert "BACK_KNEE" in codes and "FRONT_KNEE" not in codes and "ARM_SLOT" not in codes


def test_missing_joints() -> None:
    s = _settings()
    kp = _frame(_synth(), "ffc")
    hidden = {n: (None if "knee" in n or "hip" in n else k) for n, k in kp.items()}
    with pytest.raises(PoseQualityError, match="Too few body joints"):
        analyze_keypoints(hidden, 1280, 720, s)
    no_feet = {n: (None if "ankle" in n or "heel" in n or "toe" in n else k) for n, k in kp.items()}
    r = analyze_keypoints(no_feet, 1280, 720, s, phase="ffc")
    assert r.height_px is None and any("height" in w for w in r.warnings)
    assert _check(r, "FRONT_KNEE").value is None


class _FakeBackend:
    name = "fake"

    def __init__(self, kp: dict[str, Keypoint | None]) -> None:
        self.kp = kp

    def estimate(self, frame: np.ndarray, bbox: object) -> dict[str, Keypoint | None]:
        return self.kp


def test_analyze_image_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bowlsense.image_mode as im

    r = _synth()
    image = np.full((720, 1280, 3), 40, np.uint8)
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), image)
    monkeypatch.setattr(im, "detect_person", lambda img, s: (100.0, 50.0, 600.0, 700.0))
    out = tmp_path / "annotated.png"
    result = analyze_image(
        path, _settings(), phase="release", backend=_FakeBackend(_frame(r, "release")),
        annotated_out=out,
    )  # fmt: skip
    assert result.phase == "release" and result.annotated == "annotated.png"
    written = cv2.imread(str(out))
    assert written is not None and written.shape == (720, 1280, 3)


def test_image_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(InvalidInputError, match="not found"):
        read_image(tmp_path / "nope.png")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(InvalidInputError, match="not a readable"):
        read_image(bad)
    import bowlsense.image_mode as im

    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.zeros((64, 64, 3), np.uint8))

    class _NoOne:
        def predict(self, *a: object, **k: object) -> list[object]:
            class R:
                boxes = None

            return [R()]

    monkeypatch.setattr("bowlsense.detect.person._new_tracker_model", lambda w: _NoOne())
    if not (_settings().storage.models_path / _settings().detect.yolo_model).is_file():
        pytest.skip("YOLO weights not downloaded")
    with pytest.raises(NoBowlerDetectedError):
        im.detect_person(read_image(blank), _settings())
