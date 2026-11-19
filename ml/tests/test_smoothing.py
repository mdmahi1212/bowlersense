"""Tests for pose/smoothing.py."""

from __future__ import annotations

import numpy as np
import pytest

from bowlsense.config import PoseSettings, Settings
from bowlsense.errors import InvalidInputError
from bowlsense.geometry.angles import joint_angle, keypoint_track
from bowlsense.pose.smoothing import (
    clean_sequence,
    fill_short_gaps,
    savgol_window,
    smooth_runs,
)
from bowlsense.schemas import FramePose, Keypoint, PoseSequence
from synth.skeleton_gen import SynthParams, generate


def _pose(settings: Settings, **kw: object) -> PoseSettings:
    return settings.pose.model_copy(update=kw)


def _line_seq(
    values: list[float | None], confs: list[float] | None = None, fps: float = 120.0
) -> PoseSequence:
    """A sequence where only the nose moves: x = value, y = 2 * value."""
    frames = []
    for i, v in enumerate(values):
        c = 0.9 if confs is None else confs[i]
        kp = None if v is None else Keypoint(x=v, y=2 * v, z=None, conf=c)
        frames.append(FramePose(frame_idx=i, t=i / fps, keypoints={"nose": kp}))
    return PoseSequence(fps=fps, width=1000, height=1000, frames=frames)


def _nose_x(seq: PoseSequence) -> list[float | None]:
    out: list[float | None] = []
    for f in seq.frames:
        kp = f.get("nose")
        out.append(None if kp is None else kp.x)
    return out


@pytest.mark.parametrize(
    ("fps", "expected"), [(120, 13), (240, 25), (60, 7), (30, 3), (10, 3), (100, 11)]
)
def test_savgol_window_odd(fps: float, expected: int) -> None:
    assert savgol_window(fps, 0.1, 2) == expected


def test_savgol_window_validates() -> None:
    with pytest.raises(InvalidInputError):
        savgol_window(0, 0.1, 2)
    assert savgol_window(10, 0.1, 3) == 5  # needs > polyorder and odd


def test_fill_short_gaps_interior_only() -> None:
    v = np.array([np.nan, 0.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan])
    valid = np.isfinite(v)
    filled = fill_short_gaps(v, valid, max_gap=2)
    assert v[2] == pytest.approx(1.0) and v[3] == pytest.approx(2.0)
    assert not filled[0] and not filled[9]  # edges never extrapolated
    assert not filled[5:8].any()  # 3-frame gap > max_gap stays missing
    assert filled[1:5].all()


def test_fill_short_gaps_noop_cases() -> None:
    v = np.array([1.0, np.nan, 3.0])
    assert not fill_short_gaps(v.copy(), np.isfinite(v), 0)[1]
    single = np.array([np.nan, 5.0, np.nan])
    assert fill_short_gaps(single, np.isfinite(single), 5).tolist() == [False, True, False]


def test_smooth_runs_keeps_runs_separate() -> None:
    v = np.concatenate([np.zeros(20), [np.nan] * 10, np.full(20, 100.0)])
    valid = np.isfinite(v)
    out = smooth_runs(v, valid, 13, 2)
    np.testing.assert_allclose(out[:20], 0.0, atol=1e-9)  # no bleed from the 100s
    np.testing.assert_allclose(out[30:], 100.0, atol=1e-9)
    assert np.isnan(out[20:30]).all()


def test_smooth_runs_short_run_untouched_or_shrunk() -> None:
    v = np.array([0.0, 10.0, 0.0, np.nan, 1.0, 5.0, 1.0, 5.0, 1.0])
    out = smooth_runs(v, np.isfinite(v), 13, 2)
    np.testing.assert_allclose(out[:3], v[:3], atol=1e-9)  # window 3 = exact quadratic fit
    assert out[5] < 5.0  # 5-sample run is smoothed with a shrunk window


def test_smooth_preserves_polynomials() -> None:
    t = np.arange(50.0)
    v = 0.5 * t**2 - 3 * t + 7
    np.testing.assert_allclose(smooth_runs(v, np.ones(50, bool), 13, 2), v, atol=1e-6)


def test_low_confidence_dropped_and_long_gaps_kept(settings: Settings) -> None:
    vals: list[float | None] = [float(i) for i in range(30)]
    confs = [0.9] * 30
    confs[10] = 0.1  # single low-conf frame -> interpolated
    for i in range(15, 22):  # 7 missing frames > max_gap 5 -> stays missing
        vals[i] = None
    out = clean_sequence(_line_seq(vals, confs), settings.pose)
    xs = _nose_x(out)
    assert xs[10] == pytest.approx(10.0, abs=1e-6)
    nose10 = out.frames[10].get("nose")
    assert nose10 is not None and nose10.conf == pytest.approx(0.9)
    assert all(xs[i] is None for i in range(15, 22))
    assert xs[14] == pytest.approx(14.0, abs=1e-6) and xs[22] == pytest.approx(22.0, abs=1e-6)


def test_gap_at_max_is_filled(settings: Settings) -> None:
    vals: list[float | None] = [float(i) for i in range(20)]
    for i in range(5, 10):  # exactly 5 frames
        vals[i] = None
    xs = _nose_x(clean_sequence(_line_seq(vals), settings.pose))
    assert all(x is not None for x in xs)
    assert xs[7] == pytest.approx(7.0, abs=1e-6)


def test_interpolated_conf_is_lower_neighbour(settings: Settings) -> None:
    vals: list[float | None] = [0.0, 1.0, None, 3.0, 4.0]
    confs = [0.9, 0.8, 0.0, 0.6, 0.9]
    out = clean_sequence(_line_seq(vals, confs), settings.pose)
    kp = out.frames[2].get("nose")
    assert kp is not None and kp.conf == pytest.approx(0.6)


def test_all_missing_keypoints_stay_none(settings: Settings) -> None:
    out = clean_sequence(_line_seq([1.0, 2.0, 3.0]), settings.pose)
    assert all(f.get("r_toe") is None for f in out.frames)
    assert len(out.frames) == 3 and out.fps == 120


def test_cleaning_reduces_noise_on_synthetic(settings: Settings) -> None:
    """Noise and smoothing bias are checked separately: the fps/10 window must remove most
    of the noise while bending the true motion by well under 1% of body height."""
    p = SynthParams()
    clean = generate(p).sequence
    noisy = generate(p.model_copy(update={"noise_px": 2.0, "seed": 1})).sequence
    smoothed_clean = clean_sequence(clean, settings.pose)
    smoothed_noisy = clean_sequence(noisy, settings.pose)

    def rms(a: PoseSequence, b: PoseSequence, name: str) -> float:
        d = keypoint_track(a, name) - keypoint_track(b, name)
        return float(np.sqrt(np.nanmean(d**2)))

    for name in ("r_knee", "l_hip", "nose", "r_wrist"):
        raw_noise = rms(noisy, clean, name)
        noise_left = rms(smoothed_noisy, smoothed_clean, name)
        bias = rms(smoothed_clean, clean, name)
        assert noise_left < 0.5 * raw_noise, name
        assert bias < 0.005 * p.height_px, name

    def elbow(seq: PoseSequence) -> np.ndarray:
        return joint_angle(
            keypoint_track(seq, "r_shoulder"),
            keypoint_track(seq, "r_elbow"),
            keypoint_track(seq, "r_wrist"),
        )

    err_noisy = np.nanmean(np.abs(elbow(noisy) - elbow(clean)))
    err_clean = np.nanmean(np.abs(elbow(smoothed_noisy) - elbow(clean)))
    assert err_clean < 0.6 * err_noisy


def test_cleaning_fills_dropout_on_synthetic(settings: Settings) -> None:
    noisy = generate(SynthParams(dropout_rate=0.1, seed=2)).sequence
    cleaned = clean_sequence(noisy, settings.pose)
    missing_before = sum(kp is None for f in noisy.frames for kp in f.keypoints.values())
    missing_after = sum(kp is None for f in cleaned.frames for kp in f.keypoints.values())
    assert missing_before > 0
    assert missing_after < 0.02 * missing_before  # only gaps at the clip edges remain


def test_z_smoothed_when_present(settings: Settings) -> None:
    seq = generate(SynthParams(noise_px=0.0)).sequence
    out = clean_sequence(seq, settings.pose)
    kp = out.frames[100].get("r_wrist")
    assert kp is not None and kp.z is not None
    seq2 = generate(SynthParams(include_z=False)).sequence
    kp2 = clean_sequence(seq2, settings.pose).frames[100].get("r_wrist")
    assert kp2 is not None and kp2.z is None


def test_min_conf_from_settings(settings: Settings) -> None:
    strict = _pose(settings, min_conf=0.95, max_gap_frames=0)
    out = clean_sequence(_line_seq([1.0, 2.0, 3.0], [0.9, 0.99, 0.9]), strict)
    assert _nose_x(out) == [None, pytest.approx(2.0), None]
