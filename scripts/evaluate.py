"""BowlSense evaluation table for the course report.

Synthetic benchmark (ground truth known exactly; deterministic seeds):
  phase-event error (frames), robustness to 10 % keypoint dropout, bowling-arm accuracy,
  legality MAE, counter-rotation MAE, knee-collapse MAE, stride MAE, ball-speed MAE.
Real footage (if the training reports exist in data/training/reports):
  release detection (learned vs rules), bowling arm, camera view, pace/spin.
Pose PCK@0.2 (if hand-labelled keypoints exist in data/eval/pose_labels.json; format below).

Run: ml/.venv/Scripts/python scripts/evaluate.py [--quick]   (Linux: ml/.venv/bin/python)
Writes data/eval/report.json and data/eval/report.md and prints the table.

pose_labels.json: [{"video": "<path>", "frame": <int>, "keypoints": {"l_wrist": [x, y], ...}}]
evaluated against the pose backend on that frame (torso = shoulder-mid to hip-mid distance).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
OUT_DIR = ROOT / "data" / "eval"
REPORTS = ROOT / "data" / "training" / "reports"
EVENTS = ("bfc", "ffc", "arm_horizontal", "release")

Row = tuple[str, str, str, str]  # metric, value, n, note


def _quiet() -> None:
    logging.basicConfig(level=logging.ERROR)
    for name in ("bowlsense", "synth"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _mean_abs(xs: list[float]) -> float:
    return float(np.mean(np.abs(xs))) if xs else float("nan")


def phase_benchmark(quick: bool) -> tuple[list[Row], dict[str, Any]]:
    """Phase error, dropout robustness and arm accuracy over a grid of synthetic deliveries."""
    from bowlsense.config import load_settings
    from bowlsense.geometry.body import detect_bowling_arm
    from bowlsense.pipeline import analyze_delivery
    from synth.skeleton_gen import SynthParams, generate

    s = load_settings()
    seeds = range(2 if quick else 6)
    grid = [
        (arm, view, fps)
        for arm in ("right", "left")
        for view in ("side", "front")
        for fps in (120.0, 240.0)
    ]
    errors: dict[str, list[float]] = {e: [] for e in EVENTS}
    drop_err: list[float] = []
    arm_ok = n = 0
    for arm, view, fps in grid:
        for seed in seeds:
            r = generate(SynthParams(bowling_arm=arm, view=view, fps=fps, noise_px=2.0, seed=seed))
            arm_ok += detect_bowling_arm(r.sequence).arm == arm
            n += 1
            d = analyze_delivery(r.sequence, s)
            for e in EVENTS:
                errors[e].append(getattr(d.phases, e) - getattr(r.phases, e))
            rd = generate(
                SynthParams(
                    bowling_arm=arm,
                    view=view,
                    fps=fps,
                    noise_px=2.0,
                    dropout_rate=0.1,
                    seed=100 + seed,
                )
            )
            dd = analyze_delivery(rd.sequence, s)
            drop_err += [abs(getattr(dd.phases, e) - getattr(rd.phases, e)) for e in EVENTS]
    rows: list[Row] = [
        (
            f"Phase error: {e} (frames)",
            f"{_mean_abs(v):.2f} mean / {max(map(abs, v)):.0f} max",
            str(len(v)),
            "noise 2 px; 120/240 fps; side+front; L+R",
        )
        for e, v in errors.items()
    ]
    rows.append(
        (
            "Phase error with 10% dropout (frames)",
            f"{_mean_abs(drop_err):.2f} mean / {max(drop_err):.0f} max",
            str(len(drop_err)),
            "all four events",
        )
    )
    rows.append(
        ("Bowling-arm detection (synthetic)", f"{100 * arm_ok / n:.1f}%", str(n), "wrist speed")
    )
    return rows, {"phase_errors": errors, "dropout_errors": drop_err, "arm_accuracy": arm_ok / n}


def _mae_rows(
    name: str,
    unit: str,
    values: list[float],
    make: Callable[[float, int], tuple[float, float]],
    seeds: int,
    note: str,
) -> tuple[Row, list[float]]:
    errs = []
    for v in values:
        for seed in range(seeds):
            truth, measured = make(v, seed)
            errs.append(measured - truth)
    return (
        f"{name} MAE ({unit})",
        f"{_mean_abs(errs):.2f} (max {max(map(abs, errs)):.1f})",
        str(len(errs)),
        note,
    ), errs


def module_benchmark(quick: bool) -> tuple[list[Row], dict[str, Any]]:
    """Legality, alignment, knee and stride accuracy against the generator's ground truth."""
    from bowlsense.config import load_settings
    from bowlsense.pipeline import analyze_delivery
    from synth.skeleton_gen import SynthParams, generate

    s = load_settings()
    seeds = 1 if quick else 3

    def run(seed: int, **kw: Any) -> tuple[Any, Any]:
        r = generate(SynthParams(noise_px=2.0, seed=seed, **kw))
        return r, analyze_delivery(r.sequence, s)

    def legality(v: float, seed: int) -> tuple[float, float]:
        r, d = run(seed, elbow_extension_deg=v)
        return r.truth.elbow_extension_deg, d.legality.elbow_extension_deg

    def counter(v: float, seed: int) -> tuple[float, float]:
        r, d = run(seed, view="front", counter_rotation_deg=v)
        return r.truth.counter_rotation_deg, d.alignment.counter_rotation_deg or float("nan")

    def knee(v: float, seed: int) -> tuple[float, float]:
        r, d = run(seed, knee_flex_change_deg=v)
        return r.truth.knee_flex_change_deg, d.faults.measurements.get(
            "KNEE_COLLAPSE", float("nan")
        )

    def stride(v: float, seed: int) -> tuple[float, float]:
        r, d = run(seed, stride_pct=v)
        m = d.faults.measurements
        return r.truth.stride_pct, m.get("OVERSTRIDE", m.get("UNDERSTRIDE", float("nan")))

    rows, raw = [], {}
    for name, unit, values, fn, note in (
        (
            "Elbow extension (legality)",
            "deg",
            [0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
            legality,
            "side view",
        ),
        ("Counter-rotation", "deg", [10.0, 20.0, 30.0, 40.0], counter, "front view"),
        ("Front-knee collapse", "deg", [5.0, 15.0, 25.0, 35.0], knee, "side view"),
        ("Stride length", "% height", [60.0, 75.0, 90.0, 100.0], stride, "side view"),
    ):
        row, errs = _mae_rows(name, unit, values, fn, seeds, note)
        rows.append(row)
        raw[name] = errs
    legal = raw["Elbow extension (legality)"]
    return rows, {"errors": raw, "legality_n": len(legal)}


def _feeder(
    dets: dict[int, list[tuple[float, float, float]]],
) -> Callable[[object, object], list[tuple[float, float, float]]]:
    """A stand-in detector returning the next frame's prepared detections on each call."""
    frames = iter(range(len(dets)))
    return lambda _frame, _centre: dets.get(next(frames), [])


def ball_benchmark() -> tuple[list[Row], dict[str, Any]]:
    """Ball speed from a synthetic flight seen by a calibrated camera (tracker + homography)."""
    from bowlsense.analysis.delivery import assess_ball
    from bowlsense.config import load_settings
    from bowlsense.detect.ball import track_ball
    from bowlsense.geometry.pitch import calibrate
    from synth.ball_gen import ANGLED, SIDE_ON, flight

    s = load_settings()
    errs: list[float] = []
    for cam in (SIDE_ON, ANGLED):
        for fps in (120.0, 240.0):
            for speed in (80.0, 110.0, 140.0):
                world, _ = flight(speed, fps)
                px = cam.project(world)
                rng = np.random.default_rng(int(speed + fps))
                dets = {
                    i: [(float(u + rng.normal(0, 0.5)), float(v + rng.normal(0, 0.5)), 0.4)]
                    for i, (u, v) in enumerate(px)
                }
                track = track_ball(
                    ((i, np.zeros((1, 1, 3), np.uint8)) for i in range(len(px))),
                    _feeder(dets),
                    0,
                    (float(px[0][0]), float(px[0][1])),
                    fps,
                    s.ball,
                    s.delivery.first_ball_within_frames,
                )
                m = assess_ball(track, calibrate(cam.crease_points().tolist()), fps, s.delivery)
                if m.speed_kmh is not None:
                    errs.append(100.0 * (m.speed_kmh - speed) / speed)
    rows: list[Row] = [
        (
            "Ball speed error (%)",
            f"{_mean_abs(errs):.2f} (max {max(map(abs, errs)):.1f})",
            str(len(errs)),
            "calibrated; side-on + 11 deg; 0.5 px noise",
        )
    ]
    return rows, {"speed_pct_errors": errs}


def real_rows() -> tuple[list[Row], dict[str, Any]]:
    """Real-footage results from the training reports (if the dataset was used)."""
    rows: list[Row] = []
    data: dict[str, Any] = {}
    files = {p.stem: p for p in REPORTS.glob("*.json")} if REPORTS.is_dir() else {}
    if not files:
        return [("Real footage", "n/a", "0", "no training reports (dataset not used)")], data
    for stem, path in files.items():
        data[stem] = json.loads(path.read_text(encoding="utf-8"))

    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    rel = data.get("release_model", {})
    for key, label in (("learned_model", "learned BiLSTM"), ("rule_based", "rule-based")):
        part = rel.get(key)
        if isinstance(part, dict) and "within_1" in part:
            rows.append(
                (
                    f"Release within +-1 frame ({label})",
                    f"{pct(part['within_1'])} (MAE {part.get('mae_ms', float('nan')):.0f} ms)",
                    str(part.get("n", "")),
                    "real test split, 3 s crops",
                )
            )
    real = data.get("pipeline_on_real_footage", {})
    for key, label, note in (
        ("bowling_arm_elbow_at_learned_release", "elbow at learned release", "phases.use_model on"),
        ("bowling_arm", "peak wrist speed", "default rules"),
    ):
        arm = real.get(key)
        if isinstance(arm, dict) and arm.get("accuracy") is not None:
            rows.append((f"Bowling arm, real ({label})", pct(arm["accuracy"]), str(arm["n"]), note))
    learned = data.get("bowler_selector", {}).get("test_selection_accuracy")
    if isinstance(learned, dict):
        rows.append(
            (
                "Bowler picked correctly (real, learned)",
                f"{pct(learned['learned'])} (rule {pct(learned['legacy'])}, "
                f"tracker limit {pct(learned['tracker_limit'])})",
                str(data["bowler_selector"]["test_deliveries"]),
                "held-out test videos",
            )
        )
    view = data.get("pipeline_on_real_footage", {}).get("camera_view")
    if isinstance(view, dict):
        rows.append(
            ("Camera view (real)", pct(view["accuracy"]), str(view.get("n", "")), "all clips")
        )
    dt = data.get("classifiers", {}).get("delivery_type", {}).get("learned_end_to_end")
    if isinstance(dt, dict):
        rows.append(
            (
                "Pace vs spin (real)",
                f"{pct(dt['accuracy'])} acc / {pct(dt['balanced_accuracy'])} bal",
                str(dt.get("n", "")),
                "BiGRU, end to end",
            )
        )
    return rows, data


def pose_pck() -> tuple[list[Row], dict[str, Any]]:
    """PCK@0.2 of the pose backend against hand-labelled frames, if any exist."""
    labels = ROOT / "data" / "eval" / "pose_labels.json"
    if not labels.is_file():
        return [("Pose PCK@0.2", "n/a", "0", "no labels in data/eval/pose_labels.json")], {}
    from bowlsense.config import load_settings
    from bowlsense.image_mode import detect_person
    from bowlsense.io.video import read_frame
    from bowlsense.pose.base import load_backend

    s = load_settings()
    backend = load_backend(s.pose.backend, s.pose, s.storage.models_path)
    hits = total = 0
    for item in json.loads(labels.read_text(encoding="utf-8")):
        frame = read_frame(Path(item["video"]), int(item["frame"]))
        pred = backend.estimate(frame, detect_person(frame, s))
        gt = {k: np.array(v, dtype=float) for k, v in item["keypoints"].items()}
        if not all(k in gt for k in ("l_shoulder", "r_shoulder", "l_hip", "r_hip")):
            continue
        torso = np.linalg.norm(
            (gt["l_shoulder"] + gt["r_shoulder"]) / 2 - (gt["l_hip"] + gt["r_hip"]) / 2
        )
        for name, xy in gt.items():
            p = pred.get(name)
            total += 1
            hits += p is not None and np.linalg.norm(np.array([p.x, p.y]) - xy) <= 0.2 * torso
    value = f"{100 * hits / total:.1f}%" if total else "n/a"
    return [("Pose PCK@0.2", value, str(total), s.pose.backend)], {"hits": hits, "total": total}


def main(argv: list[str] | None = None) -> int:
    """Run every benchmark, print the table and write the report files."""
    parser = argparse.ArgumentParser(description="BowlSense evaluation table")
    parser.add_argument("--quick", action="store_true", help="fewer synthetic seeds (faster)")
    args = parser.parse_args(argv)
    _quiet()
    from bowlsense.offline import enable

    enable()  # the evaluation never needs the network
    rows: list[Row] = []
    raw: dict[str, Any] = {}
    benchmarks: list[tuple[str, Callable[[], tuple[list[Row], dict[str, Any]]]]] = [
        ("phases", lambda: phase_benchmark(args.quick)),
        ("modules", lambda: module_benchmark(args.quick)),
        ("ball", ball_benchmark),
        ("real", real_rows),
        ("pose", pose_pck),
    ]
    for name, fn in benchmarks:
        print(f"running {name} benchmark...", file=sys.stderr)
        r, data = fn()
        rows += r
        raw[name] = data
    widths = [max(len(r[i]) for r in [("Metric", "Value", "n", "Notes"), *rows]) for i in range(4)]
    header = ("Metric", "Value", "n", "Notes")
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
        *(
            "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) + " |"
            for r in rows
        ),
    ]
    table = "\n".join(lines)
    print(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.md").write_text(
        "# BowlSense evaluation\n\n" + table + "\n", encoding="utf-8"
    )
    (OUT_DIR / "report.json").write_text(
        json.dumps({"rows": rows, "raw": raw}, indent=1, default=float), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'report.md'} and report.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
