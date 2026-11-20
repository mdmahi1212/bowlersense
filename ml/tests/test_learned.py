"""Tests for the learned components: pose features, the BiLSTM phase/release model, the
pace-vs-spin classifier, their pipeline integration, and the training helpers. None of these
need the real dataset or trained weights."""

from __future__ import annotations

import argparse
import json
import logging
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from bowlsense.analysis.delivery_type import (
    FEATURES,
    DeliveryTypeModel,
    delivery_features,
    load_delivery_type_model,
)
from bowlsense.config import Settings, load_settings
from bowlsense.errors import InvalidInputError, ModelNotFoundError
from bowlsense.geometry.body import detect_bowling_arm_at_release, elbow_lift, estimate_height_px
from bowlsense.phases.features import FEATURE_VERSION, N_FEATURES, frame_features, role_order
from bowlsense.phases.heuristic import detect_phases
from bowlsense.phases.model import (
    PHASE_NAMES,
    build_network,
    load_phase_model,
    meta_path,
)
from bowlsense.phases.seqclf import build_classifier, load_classifier
from bowlsense.pipeline import analyze_delivery, choose_bowling_arm, classify_delivery_type
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import FramePose, Keypoint, PoseSequence
from synth.skeleton_gen import SynthParams, generate
from train.dataset import deliveries
from train.extract_poses import fill_boxes
from train.train_release import Sample, crop_bounds, eval_crops, summarise, train

logging.getLogger("bowlsense.phases.heuristic").setLevel(logging.ERROR)


@cache
def _settings() -> Settings:
    return load_settings()


def _scaled(seq: PoseSequence, k: float, dx: float) -> PoseSequence:
    frames = [
        f.model_copy(
            update={
                "keypoints": {
                    n: (kp.model_copy(update={"x": kp.x * k + dx, "y": kp.y * k}) if kp else None)
                    for n, kp in f.keypoints.items()
                }
            }
        )
        for f in seq.frames
    ]
    return seq.model_copy(update={"frames": frames})


# --- features ------------------------------------------------------------------------------------


def test_feature_shape_and_invariance() -> None:
    seq = generate(SynthParams(fps=60)).sequence
    f = frame_features(seq)
    assert f.shape == (len(seq.frames), N_FEATURES) and np.isfinite(f).all()
    g = frame_features(_scaled(seq, 1.7, 250.0))  # zoom + pan
    np.testing.assert_allclose(f, g, atol=1e-9)


def test_features_left_arm_matches_right_by_role() -> None:
    right = generate(SynthParams(fps=60)).sequence
    left = generate(SynthParams(fps=60, bowling_arm="left")).sequence
    np.testing.assert_allclose(frame_features(right), frame_features(left), atol=1e-9)
    assert role_order("right")[1] == "r_shoulder" and role_order("left")[1] == "l_shoulder"


def test_features_mark_missing_joints() -> None:
    seq = generate(SynthParams(fps=60)).sequence
    frames = list(seq.frames)
    frames[10] = frames[10].model_copy(
        update={"keypoints": {**frames[10].keypoints, "r_wrist": None}}
    )
    f = frame_features(seq.model_copy(update={"frames": frames}))
    wrist = role_order("right").index("r_wrist")
    per_joint = f.reshape(len(f), -1, 5)  # [x, y, vx, vy, visible] per joint
    assert per_joint[10, wrist, 4] == 0.0 and per_joint[11, wrist, 4] == 1.0
    assert not per_joint[10, wrist, :4].any()


def test_features_degenerate_sequence_is_zero() -> None:
    frame = FramePose(frame_idx=0, t=0.0, keypoints={"nose": Keypoint(x=1, y=1, conf=1.0)})
    seq = PoseSequence(fps=30, width=10, height=10, frames=[frame])
    assert not frame_features(seq).any()


# --- phase model ---------------------------------------------------------------------------------


def _save_model(folder: Path, hidden: int = 8) -> Path:
    import torch

    net = build_network(hidden=hidden, layers=1, dropout=0.0)
    weights = folder / "phase_model.pt"
    torch.save(net.state_dict(), weights)
    meta = {"feature_version": FEATURE_VERSION, "network": {"hidden": hidden, "layers": 1,
                                                           "dropout": 0.0}}  # fmt: skip
    meta_path(weights).write_text(json.dumps(meta), encoding="utf-8")
    load_phase_model.cache_clear()
    return weights


def test_network_shapes() -> None:
    import torch

    net = build_network(hidden=8, layers=2)
    phase, release = net(torch.zeros(3, 25, N_FEATURES))
    assert phase.shape == (3, 25, len(PHASE_NAMES)) and release.shape == (3, 25)


def test_load_and_predict(tmp_path: Path) -> None:
    weights = _save_model(tmp_path)
    model = load_phase_model(weights)
    seq = clean_sequence(generate(SynthParams(fps=60)).sequence, _settings().pose)
    pred = model.predict(seq)
    assert 0 <= pred.release < len(seq.frames)
    assert 0.0 <= pred.release_confidence <= 1.0
    assert pred.phase.shape == (len(seq.frames),)


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match=r"train\.train_release"):
        load_phase_model(tmp_path / "phase_model.pt")
    weights = _save_model(tmp_path)
    meta = json.loads(meta_path(weights).read_text(encoding="utf-8"))
    meta["feature_version"] = FEATURE_VERSION + 1
    meta_path(weights).write_text(json.dumps(meta), encoding="utf-8")
    load_phase_model.cache_clear()
    with pytest.raises(ValueError, match="feature version"):
        load_phase_model(weights)
    load_phase_model.cache_clear()


def test_training_loop_smoke() -> None:
    """A few epochs on synthetic deliveries: runs, keeps the best epoch, and learns
    something (validation error well below a random guess)."""
    samples = []
    for k in range(10):
        r = generate(SynthParams(fps=30, seed=k, noise_px=1.0))
        seq = clean_sequence(r.sequence, _settings().pose)
        n = len(seq.frames)
        phase = np.zeros(n, dtype=np.int64)
        phase[r.phases.bound : r.phases.release] = 1
        phase[r.phases.release :] = 2
        samples.append(Sample(f"s{k}", f"v{k}", seq, frame_features(seq), phase, r.phases.release))
    args = argparse.Namespace(seed=0, hidden=16, layers=1, dropout=0.0, lr=5e-3, epochs=25,
                              patience=25, batch=4, noise=0.0, sigma=1.0,
                              release_weight=2.0)  # fmt: skip
    _net, info = train(samples[:8], samples[8:], args)
    assert len(info["history"]) == 25
    assert info["best_val_mae_frames"] < 20.0  # random guess over a ~90-frame crop is ~30


def test_crops_place_release_inside() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(40, 200))
        release = int(rng.integers(6, n - 6))
        a, z = crop_bounds(n, release, int(rng.integers(20, 120)), rng)
        assert 0 <= a <= release < z <= n


def test_summarise() -> None:
    m = summarise([0, 1, -2, 4, None], fps=30.0)
    assert m["within_1"] == pytest.approx(0.4) and m["within_3"] == pytest.approx(0.6)
    assert m["no_answer"] == pytest.approx(0.2) and m["mae_frames"] == pytest.approx(7 / 4)


def test_eval_crops_deterministic() -> None:
    seq = clean_sequence(generate(SynthParams(fps=30)).sequence, _settings().pose)
    s = Sample("k", "v", seq, frame_features(seq), np.zeros(len(seq.frames), np.int64), 80)
    a = [(x, y) for _, x, y in eval_crops([s], 30.0, seed=1)]
    b = [(x, y) for _, x, y in eval_crops([s], 30.0, seed=1)]
    assert a == b and len(a) == 3


# --- delivery type ------------------------------------------------------------------------------


def _type_model(coef: list[float]) -> DeliveryTypeModel:
    k = len(FEATURES)
    return DeliveryTypeModel(feature_names=list(FEATURES), means=[0.0] * k, scales=[1.0] * k,
                             coef=coef, intercept=0.0)  # fmt: skip


def test_delivery_features_on_synthetic() -> None:
    r = generate(SynthParams(fps=60))
    seq = clean_sequence(r.sequence, _settings().pose)
    f = delivery_features(seq, r.phases.release)
    assert f.shape == (len(FEATURES),) and np.isfinite(f).all()
    assert f[0] > 5.0  # the bowling wrist moves several torso lengths per second at release
    assert np.isnan(delivery_features(seq, 10_000)).all()


def test_delivery_type_model_logic_and_json(tmp_path: Path) -> None:
    k = len(FEATURES)
    model = _type_model([1.0] + [0.0] * (k - 1))
    fast = np.zeros(k)
    fast[0] = 3.0
    assert model.p_fast(fast) > 0.9 and model.p_fast(-fast) < 0.1
    assert model.p_fast(np.full(k, np.nan)) == pytest.approx(0.5)  # NaN -> training mean
    path = tmp_path / "delivery_type.json"
    path.write_text(model.model_dump_json(), encoding="utf-8")
    load_delivery_type_model.cache_clear()
    assert load_delivery_type_model(path) == model
    bad = model.model_copy(update={"feature_names": ["x"]})
    with pytest.raises(ValueError, match="retrain"):
        bad.p_fast(fast)
    with pytest.raises(ModelNotFoundError, match=r"train\.train_delivery_type"):
        load_delivery_type_model(tmp_path / "missing.json")


# --- pipeline integration -------------------------------------------------------------------------


def _settings_with_models(folder: Path, use_model: bool) -> Settings:
    s = _settings()
    return s.model_copy(
        update={
            "storage": s.storage.model_copy(update={"models_dir": folder}),
            "phases": s.phases.model_copy(update={"use_model": use_model}),
        }
    )


def _save_classifier(folder: Path, name: str = "delivery_type_seq") -> Path:
    import torch

    net = build_classifier(2, hidden=8, dropout=0.0)
    weights = folder / f"{name}.pt"
    torch.save(net.state_dict(), weights)
    meta = {"feature_version": FEATURE_VERSION, "classes": ["spin", "fast"], "layout": "role",
            "network": {"hidden": 8, "dropout": 0.0}}  # fmt: skip
    weights.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    load_classifier.cache_clear()
    return weights


def test_pipeline_uses_learned_models_when_enabled(tmp_path: Path) -> None:
    _save_model(tmp_path)
    _save_classifier(tmp_path)
    r = generate(SynthParams())
    on = analyze_delivery(r.sequence, _settings_with_models(tmp_path, True))
    assert on.release_source == "model"
    assert on.delivery_type is not None and on.delivery_type.delivery_type in ("fast", "spin")
    assert 0.0 <= on.delivery_type.p_fast <= 1.0
    off = analyze_delivery(r.sequence, _settings_with_models(tmp_path / "empty", False))
    assert off.release_source == "rules" and off.delivery_type is None


def test_sequence_classifier_round_trip(tmp_path: Path) -> None:
    weights = _save_classifier(tmp_path)
    clf = load_classifier(weights)
    seq = clean_sequence(generate(SynthParams(fps=30)).sequence, _settings().pose)
    probs = clf.predict_proba(seq)
    assert set(probs) == {"spin", "fast"} and sum(probs.values()) == pytest.approx(1.0)
    assert clf.predict(seq)[0] in ("spin", "fast")
    assert classify_delivery_type(seq, 50, tmp_path / "missing.pt") is None
    with pytest.raises(ModelNotFoundError, match=r"train\.train_classifiers"):
        load_classifier(tmp_path / "missing.pt")
    meta = json.loads(weights.with_suffix(".json").read_text(encoding="utf-8"))
    meta["feature_version"] = FEATURE_VERSION + 1
    weights.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    load_classifier.cache_clear()
    with pytest.raises(ValueError, match="feature version"):
        load_classifier(weights)
    load_classifier.cache_clear()


# --- bowling arm at release ------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["right", "left"])
@pytest.mark.parametrize("view", ["side", "front", "behind"])
def test_elbow_at_true_release_finds_arm(arm: str, view: str) -> None:
    r = generate(SynthParams.model_validate({"bowling_arm": arm, "view": view, "noise_px": 2.0}))
    seq = clean_sequence(r.sequence, _settings().pose)
    result = detect_bowling_arm_at_release(seq, r.phases.release)
    assert result.arm == arm and result.method == "elbow_at_release"
    assert result.elbow_margin_px is not None and result.elbow_margin_px > 5.0
    assert elbow_lift(seq, arm, r.phases.release) > 0  # bowling elbow is the higher one


def test_choose_bowling_arm_order(tmp_path: Path) -> None:
    seq = clean_sequence(generate(SynthParams(bowling_arm="left")).sequence, _settings().pose)
    profile = choose_bowling_arm(seq, _settings(), known="right")
    assert profile.arm == "right" and profile.method == "profile"
    rules = choose_bowling_arm(seq, _settings_with_models(tmp_path / "none", True))
    assert rules.method == "wrist_speed" and rules.arm == "left"
    _save_model(tmp_path)
    learned = choose_bowling_arm(seq, _settings_with_models(tmp_path, True))
    assert learned.method in ("elbow_at_release", "wrist_speed")


def test_elbow_lift_missing_is_nan_and_falls_back() -> None:
    r = generate(SynthParams())
    frames = [
        f.model_copy(update={"keypoints": {**f.keypoints, "l_elbow": None, "r_elbow": None}})
        for f in r.sequence.frames
    ]
    seq = r.sequence.model_copy(update={"frames": frames})
    assert np.isnan(elbow_lift(seq, "right", r.phases.release))
    assert detect_bowling_arm_at_release(seq, r.phases.release).method == "wrist_speed"


def test_detect_phases_release_override() -> None:
    s = _settings()
    r = generate(SynthParams())
    seq = clean_sequence(r.sequence, s.pose)
    h = estimate_height_px(seq, s.body).height_px
    forced = detect_phases(seq, h, s.phases, release=r.phases.release + 1)
    assert forced.release == r.phases.release + 1
    with pytest.raises(InvalidInputError, match="outside"):
        detect_phases(seq, h, s.phases, release=10_000)


# --- dataset reader ------------------------------------------------------------------------


def test_dataset_reader_on_tiny_fake(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data_split.csv").write_text(
        "video_name,camera_angle,number_bowl_release_events,set_name\n"
        "v1,mid_mid_left,1,train\nv2,far_mid_right,0,challenge\n",
        encoding="utf-8",
    )
    na = {"bowling_type": "not applicable", "bowling_from": "not applicable",
          "bowler_hand": "not applicable", "box": [0, 0, 1, 1, 0]}  # fmt: skip
    events: dict[str, list[dict[str, object]]] = {
        str(f): [{"event": "is bowling", **na}] for f in range(10, 15)
    }
    events["15"] = [{"event": "bowl release", "bowling_type": "fast bowler",
                     "bowling_from": "over the wicket", "bowler_hand": "left-handed",
                     "box": [0, 0, 1, 1, 0]}]  # fmt: skip
    ann = {
        "metadata": {"video_name": "v1"},
        "event": events,
        "scene": {"15": [{"bowl_direction": "towards camera"}]},
        "person": {"12": [{"role": "bowler", "id": "1", "box": [10, 20, 30, 60, 0]}]},
    }
    (tmp_path / "data" / "v1.json").write_text(json.dumps(ann), encoding="utf-8")
    [d] = deliveries(tmp_path)
    assert (d.span_start, d.release, d.split) == (10, 15, "train")
    assert (d.bowling_type, d.bowler_hand, d.bowling_from) == ("fast", "left", "over")
    assert d.bowl_direction == "towards camera" and d.key == "v1__000015"


def test_fill_boxes_interpolates_short_gaps() -> None:
    boxes = {0: (0.0, 0.0, 10.0, 10.0), 4: (4.0, 0.0, 14.0, 10.0), 30: (0.0, 0.0, 1.0, 1.0)}
    out = fill_boxes(boxes, 0, 30, max_gap=5)
    assert out[2] == pytest.approx((2.0, 0.0, 12.0, 10.0))
    assert 10 not in out  # 26-frame gap is too long to bridge
