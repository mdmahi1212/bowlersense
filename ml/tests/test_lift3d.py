"""Phase 15: optional 3D lifting behind pose.lift_3d (MotionBERT ONNX)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from bowlsense.config import load_settings
from bowlsense.errors import ModelNotFoundError
from bowlsense.pose.lift3d import H36M, lift_sequence, load_lifter, to_h36m, windows
from synth.skeleton_gen import SynthParams, generate


def _export_fake_motionbert(path: Path, window: int) -> Path:
    """A stand-in network with MotionBERT's input/output shapes: z = 0.5 * x."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")

    class Fake(torch.nn.Module):  # type: ignore[misc,name-defined]
        def forward(self, x: object) -> object:
            return torch.cat([x[..., :2], 0.5 * x[..., :1]], dim=-1)  # type: ignore[index]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(Fake(), (torch.zeros(1, window, 17, 3),), str(path),
                          input_names=["keypoints_2d"], output_names=["keypoints_3d"],
                          dynamo=False)  # fmt: skip
    return path


def test_h36m_mapping_and_windows() -> None:
    seq = generate(SynthParams(height_px=300.0)).sequence
    x = to_h36m(seq)
    assert x.shape == (len(seq.frames), 17, 3) and len(H36M) == 17
    f = seq.frames[10].keypoints
    scale = min(seq.width, seq.height) / 2
    rw = f["r_wrist"]
    assert rw is not None
    assert x[10, 16, 0] == pytest.approx((rw.x - seq.width / 2) / scale, abs=1e-5)
    assert np.all(np.abs(x[..., :2]) < 5) and np.all((x[..., 2] >= 0) & (x[..., 2] <= 1))
    assert windows(250, 243) == [(0, 243), (243, 250)] and windows(0, 243) == []


def test_flag_off_by_default_and_missing_model_is_clear(tmp_path: Path) -> None:
    s = load_settings()
    assert s.pose.lift_3d is False  # the system works fully without 3D lifting
    with pytest.raises(ModelNotFoundError, match="export_motionbert"):
        load_lifter(tmp_path / "motionbert_lite.onnx", s.pose.lift_window)


def test_lift_fills_z_relative_to_hips(tmp_path: Path) -> None:
    window = 27
    model = _export_fake_motionbert(tmp_path / "fake.onnx", window)
    seq = generate(SynthParams(height_px=300.0, dropout_rate=0.05)).sequence
    lifted = lift_sequence(seq, load_lifter(model, window))
    assert len(lifted.frames) == len(seq.frames)
    i = 50
    f2, f3 = seq.frames[i].keypoints, lifted.frames[i].keypoints
    lh, rh, rw = f2["l_hip"], f2["r_hip"], f3["r_wrist"]
    if lh is not None and rh is not None and rw is not None and f2["r_wrist"] is not None:
        # fake net: z = 0.5 * scaled x  ->  pixels relative to the hip centre
        assert rw.z == pytest.approx(0.5 * (f2["r_wrist"].x - (lh.x + rh.x) / 2), abs=1e-2)
        assert (rw.x, rw.y) == (f2["r_wrist"].x, f2["r_wrist"].y)
    assert all(k is None or k.z is not None for f in lifted.frames for k in f.keypoints.values())
    missing = [
        (j, n) for j, f in enumerate(seq.frames) for n, k in f.keypoints.items() if k is None
    ]
    assert all(lifted.frames[j].keypoints[n] is None for j, n in missing)


def test_pipeline_lifts_only_when_flag_on(tmp_path: Path) -> None:
    from bowlsense.pipeline import extract_pose
    from synth.skeleton_gen import render_to_video

    s = load_settings()
    r = generate(SynthParams(fps=120.0, height_px=300.0))
    video = render_to_video(r.sequence, tmp_path / "v.mp4")
    pose = tmp_path / "pose.json"
    pose.write_text(r.sequence.model_dump_json(), encoding="utf-8")
    off = extract_pose(video, s, pose_json=pose)
    assert [f.keypoints for f in off.sequence.frames] == [f.keypoints for f in r.sequence.frames]
    models = tmp_path / "models"
    models.mkdir()
    _export_fake_motionbert(models / s.pose.lift_model, s.pose.lift_window)
    on = s.model_copy(update={
        "pose": s.pose.model_copy(update={"lift_3d": True}),
        "storage": s.storage.model_copy(update={"models_dir": models}),
    })  # fmt: skip
    lifted = extract_pose(video, on, pose_json=pose)
    f_in, f_out = r.sequence.frames[60].keypoints, lifted.sequence.frames[60].keypoints
    rw_in, rw_out, lh, rh = f_in["r_wrist"], f_out["r_wrist"], f_in["l_hip"], f_in["r_hip"]
    assert rw_in is not None and rw_out is not None and lh is not None and rh is not None
    assert rw_out.z == pytest.approx(0.5 * (rw_in.x - (lh.x + rh.x) / 2), abs=1e-2)
    assert (rw_out.x, rw_out.y) == (rw_in.x, rw_in.y)
