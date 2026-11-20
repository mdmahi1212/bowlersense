"""The evaluation suite behind the "Model evaluation" page and the course-report tables.

Classification tasks get a confusion matrix, accuracy, balanced accuracy, precision, recall,
F1 (per class, macro and weighted) and the majority-class baseline; measurements get MAE,
RMSE, R^2 and bias. Every task says whether it was measured on **real** labelled footage (the
dataset's held-out test videos) or on **synthetic** bowlers with exactly known answers (for
things the dataset has no labels for: legality, faults, joint angles).

Run: ml/.venv/Scripts/python -m train.eval_suite [--quick]
Writes data/eval/metrics.json, data/eval/evaluation.md and data/eval/figures/*.png.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.analysis.biomech import biomechanics
from bowlsense.analysis.faults import FAULT_SPECS
from bowlsense.config import Settings, load_settings
from bowlsense.evaluation import (
    ClassificationReport,
    EvaluationReport,
    RegressionReport,
    ScoreItem,
    classification_report,
    regression_report,
)
from bowlsense.pipeline import analyze_delivery
from bowlsense.pose.smoothing import clean_sequence
from synth.skeleton_gen import SynthParams, SynthResult, generate

log = logging.getLogger("train.eval_suite")
SPEC = {spec.code: spec for spec in FAULT_SPECS}
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "eval"
REPORTS = ROOT / "data" / "training" / "reports"


# --- synthetic -----------------------------------------------------------------------------------


def _analyse(params: SynthParams, s: Settings) -> tuple[SynthResult, Any]:
    r = generate(params)
    return r, analyze_delivery(r.sequence, s)


def eval_legality(s: Settings, quick: bool) -> ClassificationReport:
    """Legal vs not legal on synthetic bowlers with a known elbow extension."""
    limit = s.legality.elbow_extension_threshold_deg
    values = [0, 4, 8, 11, 13, 17, 19, 22, 26, 30]
    seeds = range(1 if quick else 2)
    truth, pred = [], []
    for ext in values:
        for arm in ("right", "left"):
            for fps in (120.0, 240.0):
                for seed in seeds:
                    r, d = _analyse(SynthParams(elbow_extension_deg=ext, bowling_arm=arm, fps=fps,
                                                noise_px=2.0, seed=seed), s)  # fmt: skip
                    truth.append("not_legal" if r.truth.elbow_extension_deg > limit else "legal")
                    v = d.legality.verdict
                    pred.append(None if v == "cannot_judge" else v)
    for view in ("front",):  # the system should decline these, not guess
        for ext in (8, 22):
            r, d = _analyse(SynthParams(elbow_extension_deg=ext, view=view, fps=240.0), s)
            truth.append("not_legal" if r.truth.elbow_extension_deg > limit else "legal")
            pred.append(None if d.legality.verdict == "cannot_judge" else d.legality.verdict)
    return classification_report(
        "Legality (elbow extension)",
        f"Legal vs not legal (> {limit:.0f} deg) on synthetic side-on bowlers with known elbow "
        "extension (0-30 deg, both arms, 120/240 fps, 2 px keypoint noise). Front-on clips are "
        "included to check the system declines them ('can't judge' = abstained).",
        "synthetic", truth, pred, ["legal", "not_legal"],
    )  # fmt: skip


FAULT_SWEEPS: dict[str, tuple[str, list[float], dict[str, Any]]] = {
    "KNEE_COLLAPSE": ("knee_flex_change_deg", [4, 10, 15, 18, 22, 26, 32, 40], {}),
    "OVERSTRIDE": ("stride_pct", [70, 78, 84, 88, 92, 96, 100, 104], {}),
    "UNDERSTRIDE": ("stride_pct", [60, 62, 64, 67, 70, 75, 80, 85], {}),
    "RUNUP_DECEL": ("runup_decel_pct", [0, 5, 10, 13, 18, 22, 28, 35], {}),
    "LOW_RELEASE": ("release_height_pct", [105, 108, 111, 113, 117, 120, 123, 125], {}),
    "SHORT_FOLLOW": ("follow_through_pct", [15, 25, 32, 37, 45, 55, 70, 90], {}),
    "WEAK_FRONT_ARM": ("front_arm_ffc_deg", [-60, -40, -20, -5, 10, 30, 60, 90], {}),
    "CHAIN_ORDER": ("pelvis_rotation_delay_s", [0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.15, 0.2], {}),
    "LATERAL_FLEXION": ("lateral_flexion_deg", [5, 15, 22, 27, 33, 38, 44, 50], {"view": "front"}),
    "HEAD_FALL": ("lateral_flexion_deg", [5, 15, 22, 27, 33, 38, 44, 50], {"view": "front"}),
}


def _fault_truth(code: str, r: SynthResult, s: Settings) -> bool:
    """Whether the fault is really present, from the generator's ground truth."""
    t, f = r.truth, s.faults
    if code == "CHAIN_ORDER":
        p, tr, a = t.chain_peak_times_s
        return bool(max(0.0, p - tr, tr - a, p - a) > SPEC[code].threshold(f))
    value = {
        "KNEE_COLLAPSE": t.knee_flex_change_deg, "OVERSTRIDE": t.stride_pct,
        "UNDERSTRIDE": t.stride_pct, "RUNUP_DECEL": t.runup_decel_pct,
        "LOW_RELEASE": t.release_height_pct, "SHORT_FOLLOW": t.follow_through_pct,
        "WEAK_FRONT_ARM": -t.front_wrist_above_shoulder_ffc_pct,
        "LATERAL_FLEXION": t.trunk_tilt_front_deg, "HEAD_FALL": t.head_offset_pct,
    }[code]  # fmt: skip
    threshold = SPEC[code].threshold(f)
    below = code in ("UNDERSTRIDE", "LOW_RELEASE", "SHORT_FOLLOW")
    return bool(value < threshold if below else value > threshold)


def eval_faults(s: Settings, quick: bool) -> list[ClassificationReport]:
    """One present/absent report per technique fault, plus all faults pooled."""
    out = []
    pooled_t: list[str] = []
    pooled_p: list[str | None] = []
    for code, (param, values, extra) in FAULT_SWEEPS.items():
        truth: list[str] = []
        pred: list[str | None] = []
        for v in values:
            for seed in range(1 if quick else 2):
                params = SynthParams.model_validate({param: v, "noise_px": 2.0, "seed": seed,
                                                     **extra})  # fmt: skip
                try:
                    r, d = _analyse(params, s)
                except ValueError:  # not physically reachable with the other defaults
                    continue
                truth.append("present" if _fault_truth(code, r, s) else "absent")
                if code in d.faults.not_measured:
                    pred.append(None)
                else:
                    pred.append(
                        "present" if code in {x.code for x in d.faults.faults} else "absent"
                    )
        rep = classification_report(
            f"Fault: {SPEC[code].name}",
            f"{code} detected or not on synthetic bowlers, sweeping {param} across the threshold "
            f"({extra.get('view', 'side')} view, 2 px noise).",
            "synthetic", truth, pred, ["absent", "present"],
        )  # fmt: skip
        out.append(rep)
        pooled_t += truth
        pooled_p += pred
    out.insert(0, classification_report(
        "Faults (all 10 pooled)", "Every fault-detection case above pooled into one matrix.",
        "synthetic", pooled_t, pooled_p, ["absent", "present"],
    ))  # fmt: skip
    return out


# (biomech key, truth getter, unit, view, description)
BIOMECH_TRUTH: list[tuple[str, Any, str, str, str]] = [
    ("knee_flexion_change", lambda t: t.knee_flex_change_deg, "deg", "side",
     "Front-knee flexion from FFC to release"),
    ("stride", lambda t: t.stride_pct, "% of height", "side", "Delivery stride length"),
    ("release_height", lambda t: t.release_height_pct, "% of height", "side", "Release height"),
    ("runup_decel", lambda t: t.runup_decel_pct, "%", "side", "Run-up speed loss"),
    ("follow_through", lambda t: t.follow_through_pct, "% of stride", "side", "Follow-through"),
    ("front_arm_height_ffc", lambda t: t.front_wrist_above_shoulder_ffc_pct, "% of height",
     "side", "Front-arm height at FFC"),
    ("counter_rotation", lambda t: t.counter_rotation_deg, "deg", "front",
     "Shoulder counter-rotation"),
    ("hip_shoulder_sep_ffc", lambda t: t.hip_shoulder_sep_ffc_deg, "deg", "front",
     "Hip-shoulder separation at FFC"),
    ("shoulder_alignment_bfc", lambda t: t.shoulder_yaw_bfc_deg, "deg", "front",
     "Shoulder alignment at BFC"),
    ("hip_alignment_bfc", lambda t: t.hip_yaw_bfc_deg, "deg", "front", "Hip alignment at BFC"),
    ("lateral_flexion", lambda t: t.trunk_tilt_front_deg, "deg", "front",
     "Trunk side-bend at release"),
    ("head_offset", lambda t: t.head_offset_pct, "% of height", "front",
     "Head offset from front foot"),
]  # fmt: skip


def _random_params(rng: np.random.Generator, view: str, seed: int) -> SynthParams:
    return SynthParams(
        view=view,
        bowling_arm=str(rng.choice(["right", "left"])),
        fps=float(rng.choice([120.0, 240.0])),
        elbow_extension_deg=float(rng.uniform(0, 30)),
        counter_rotation_deg=float(rng.uniform(5, 45)),
        knee_flex_change_deg=float(rng.uniform(0, 40)),
        stride_pct=float(rng.uniform(62, 100)),
        lateral_flexion_deg=float(rng.uniform(5, 45)),
        release_height_pct=float(rng.uniform(106, 125)),
        runup_decel_pct=float(rng.uniform(0, 30)),
        follow_through_pct=float(rng.uniform(25, 100)),
        front_arm_ffc_deg=float(rng.uniform(-40, 90)),
        noise_px=2.0,
        seed=seed,
    )


def eval_biomechanics(s: Settings, quick: bool) -> list[RegressionReport]:
    """Measured vs true values on randomised synthetic bowlers (side and front views)."""
    n = 12 if quick else 30
    truth: dict[str, list[float]] = {k: [] for k, *_ in BIOMECH_TRUTH}
    meas: dict[str, list[float | None]] = {k: [] for k, *_ in BIOMECH_TRUTH}
    ext_t: list[float] = []
    ext_m: list[float | None] = []
    rng = np.random.default_rng(7)
    for view in ("side", "front"):
        for i in range(n):
            while True:  # some random combinations aren't physically reachable: redraw
                try:
                    r, d = _analyse(_random_params(rng, view, i), s)
                    break
                except ValueError:
                    continue
            seq = clean_sequence(r.sequence, s.pose).model_copy(update={"view": view})
            metrics = {m.key: m.value for m in biomechanics(
                seq, d.phases, d.height_px, d.alignment, s,
                clean_sequence(r.sequence, s.pose, smooth=False))}  # fmt: skip
            for key, get, _, v, _ in BIOMECH_TRUTH:
                if v == view:
                    truth[key].append(float(get(r.truth)))
                    meas[key].append(metrics.get(key))
            if view == "side":
                ext_t.append(r.truth.elbow_extension_deg)
                ext_m.append(d.legality.elbow_extension_deg)
    out = [regression_report(
        "Elbow extension", "Elbow straightening measured by the legality check", "synthetic",
        "deg", ext_t, ext_m)]  # fmt: skip
    for key, _, unit, view, desc in BIOMECH_TRUTH:
        if any(m is not None for m in meas[key]):
            out.append(regression_report(desc, f"{desc} ({view} view)", "synthetic", unit,
                                         truth[key], meas[key]))  # fmt: skip
    return out


# --- real footage ------------------------------------------------------------------------------


def eval_real(s: Settings) -> tuple[list[ClassificationReport], list[ScoreItem], list[str]]:
    """Pace/spin, bowling arm and camera view on the dataset's labelled clips (cached poses
    from ``train.extract_poses``), plus release and bowler-pick scores from the training
    reports. Skipped with a note when the dataset hasn't been processed on this machine."""
    from bowlsense.geometry.body import detect_bowling_arm, detect_bowling_arm_at_release
    from bowlsense.io.quality import estimate_view
    from bowlsense.phases.model import load_phase_model
    from bowlsense.pipeline import classify_delivery_type
    from train.evaluate_pipeline import VIEW_OF_DIRECTION
    from train.extract_poses import DEFAULT_OUT, load_extracted

    notes: list[str] = []
    reports: list[ClassificationReport] = []
    scores: list[ScoreItem] = []
    models = s.storage.models_path
    try:
        items = load_extracted(DEFAULT_OUT)
    except (FileNotFoundError, OSError, ValueError):
        items = []
    if not items:
        notes.append("Real-footage classification skipped: no extracted dataset poses here "
                     "(run train.extract_poses with the dataset).")  # fmt: skip
    else:
        phase_model = load_phase_model(models / s.phases.model_file)
        type_t, type_p, arm_t, arm_rule, arm_learned, view_t, view_p = [], [], [], [], [], [], []
        for it in items:
            d = it.delivery
            seq = clean_sequence(it.sequence, s.pose)
            truth_view = VIEW_OF_DIRECTION.get(d.bowl_direction or "")
            if truth_view is not None and d.bowler_hand is not None:
                arm_seq = seq.model_copy(update={"bowling_arm": d.bowler_hand})
                stop = max(2, it.release_idx - round(s.body.runup_end_before_peak_s * seq.fps))
                view_t.append(truth_view)
                view_p.append(estimate_view(arm_seq, (0, stop), s.video))
            if d.split != "test" or d.bowler_hand is None:
                continue
            best = max(
                (phase_model.predict(seq.model_copy(update={"bowling_arm": a}))
                 for a in ("right", "left")),
                key=lambda p: p.release_confidence,
            )  # fmt: skip
            learned_arm = detect_bowling_arm_at_release(seq, best.release).arm
            arm_t.append(d.bowler_hand)
            arm_learned.append(learned_arm)
            arm_rule.append(detect_bowling_arm(seq).arm)
            if d.bowling_type is not None:
                live = seq.model_copy(update={"bowling_arm": learned_arm})
                result = classify_delivery_type(
                    live, phase_model.predict(live).release, models / s.delivery.type_model_file
                )
                type_t.append(d.bowling_type)
                type_p.append(None if result is None else result.delivery_type)
        if type_t:
            reports.append(classification_report(
                "Pace vs spin", "Learned sequence classifier, end to end (learned release and "
                "arm), on the dataset's held-out test videos.", "real", type_t, type_p,
                ["fast", "spin"]))  # fmt: skip
        if arm_t:
            reports.append(classification_report(
                "Bowling arm (elbow at learned release)", "The app's method when the learned "
                "phase model is on; held-out test videos.", "real", arm_t, arm_learned,
                ["right", "left"]))  # fmt: skip
            reports.append(classification_report(
                "Bowling arm (peak wrist speed)", "The default rule; held-out test videos.",
                "real", arm_t, arm_rule, ["right", "left"]))  # fmt: skip
        if view_t:
            reports.append(classification_report(
                "Camera view", "Estimated from shoulder width and size change in the run-up "
                "(rule-based, so all labelled clips are used).", "real", view_t, view_p,
                ["side", "front", "behind"]))  # fmt: skip

    def report(name: str) -> dict[str, Any]:
        path = REPORTS / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    rel = report("release_model")
    for key, label in (("learned_model", "learned BiLSTM"), ("rule_based", "rules")):
        part = rel.get(key)
        if isinstance(part, dict) and "within_1" in part:
            n = int(part.get("n", 0))
            for tol in (1, 2, 5):
                scores.append(ScoreItem(task=f"Release timing ({label})",
                                        metric=f"within +-{tol} frame", value=part[f"within_{tol}"],
                                        n=n, data="real", note="held-out test videos"))  # fmt: skip
            scores.append(ScoreItem(task=f"Release timing ({label})", metric="MAE (ms)",
                                    value=part["mae_ms"], n=n, data="real"))  # fmt: skip
    sel = report("bowler_selector").get("test_selection_accuracy")
    if isinstance(sel, dict):
        n = int(report("bowler_selector").get("test_deliveries", 0))
        for key, label in (("learned", "learned selector"), ("legacy", "old rule"),
                           ("tracker_limit", "tracker ceiling")):  # fmt: skip
            scores.append(ScoreItem(task="Bowler picked correctly", metric=label,
                                    value=sel[key], n=n, data="real",
                                    note="labelled bowler boxes, IoU >= 0.5"))  # fmt: skip
    return reports, scores, notes


# --- output ------------------------------------------------------------------------------------


def write_figures(report: EvaluationReport, folder: Path) -> None:
    """A confusion-matrix heatmap per classifier and a scatter per measurement."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(report.classification):
        m = np.array(c.confusion)
        fig, ax = plt.subplots(figsize=(3.2 + 0.4 * len(c.labels), 3.0 + 0.4 * len(c.labels)))
        ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(len(c.labels)), c.labels, rotation=30, ha="right")
        ax.set_yticks(range(len(c.labels)), c.labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for (r, k), v in np.ndenumerate(m):
            ax.text(k, r, str(v), ha="center", va="center",
                    color="white" if v > m.max() / 2 else "black")  # fmt: skip
        ax.set_title(f"{c.task}\nacc {c.accuracy:.0%}, macro F1 {c.macro_f1:.2f}", fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / f"confusion_{i:02d}.png", dpi=130)
        plt.close(fig)
    for i, r in enumerate(report.regression):
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        ax.scatter(r.truth, r.measured, s=12)
        lo, hi = min(r.truth + r.measured), max(r.truth + r.measured)
        ax.plot([lo, hi], [lo, hi], color="grey", linewidth=1)
        ax.set_xlabel(f"true ({r.unit})")
        ax.set_ylabel(f"measured ({r.unit})")
        r2 = "n/a" if r.r2 is None else f"{r.r2:.2f}"
        ax.set_title(f"{r.task}\nMAE {r.mae:.2f}, R² {r2}", fontsize=8)
        fig.tight_layout()
        fig.savefig(folder / f"regression_{i:02d}.png", dpi=130)
        plt.close(fig)


def markdown(report: EvaluationReport) -> str:
    """The report as Markdown tables."""
    lines = ["# BowlSense model evaluation", "", f"Generated {report.generated:%Y-%m-%d %H:%M}.",
             ""]  # fmt: skip
    lines += [
        "## Classification",
        "",
        "| Task | Data | n | Accuracy | Balanced acc. | Macro P | Macro R | Macro F1 | "
        "Weighted F1 | Baseline | Coverage |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report.classification:
        base = "" if c.baseline_accuracy is None else f"{c.baseline_accuracy:.1%}"
        lines.append(
            f"| {c.task} | {c.data} | {c.n} | {c.accuracy:.1%} | {c.balanced_accuracy:.1%} | "
            f"{c.macro_precision:.3f} | {c.macro_recall:.3f} | {c.macro_f1:.3f} | "
            f"{c.weighted_f1:.3f} | {base} | {c.coverage:.0%} |"
        )
    lines += ["", "### Confusion matrices (rows = true, columns = predicted)", ""]
    for c in report.classification:
        lines += [f"**{c.task}**", "", "| true \\ predicted | " + " | ".join(c.labels) + " |",
                  "|---|" + "---|" * len(c.labels)]  # fmt: skip
        for label, row in zip(c.labels, c.confusion, strict=True):
            lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
        lines.append("")
    lines += ["## Measurement accuracy", "",
              "| Measurement | Data | n | MAE | RMSE | R² | Bias | Max error | Unit |",
              "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in report.regression:
        r2 = "n/a" if r.r2 is None else f"{r.r2:.3f}"
        lines.append(f"| {r.task} | {r.data} | {r.n} | {r.mae:.2f} | {r.rmse:.2f} | {r2} | "
                     f"{r.bias:+.2f} | {r.max_abs_error:.2f} | {r.unit} |")  # fmt: skip
    lines += ["", "## Other scores", "", "| Task | Metric | Value | n | Data |",
              "|---|---|---|---|---|"]  # fmt: skip
    for sc in report.scores:
        value = f"{sc.value:.1f}" if sc.metric.startswith("MAE") else f"{sc.value:.1%}"
        lines.append(f"| {sc.task} | {sc.metric} | {value} | {sc.n} | {sc.data} |")
    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]
    return "\n".join(lines) + "\n"


def run(quick: bool = False, out: Path = OUT) -> EvaluationReport:
    """Run every evaluation and write metrics.json, evaluation.md and the figures."""
    s = load_settings()
    log.info("legality...")
    classification = [eval_legality(s, quick)]
    log.info("faults...")
    classification += eval_faults(s, quick)
    log.info("real footage...")
    real, scores, notes = eval_real(s)
    classification = real + classification
    log.info("biomechanics...")
    regression = eval_biomechanics(s, quick)
    report = EvaluationReport(
        generated=datetime.now(UTC), classification=classification, regression=regression,
        scores=scores,
        notes=[
            *notes,
            "Synthetic = bowlers from the built-in generator with exactly known answers (no "
            "real labels exist for legality, faults or joint angles). Real = the labelled "
            "match-footage dataset's held-out test videos.",
            "Legality is 2D video screening, not the ICC 3D test; injury risk is a rule-based "
            "estimate, not medical advice.",
        ],
    )  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(report.model_dump_json(indent=1), encoding="utf-8")
    (out / "evaluation.md").write_text(markdown(report), encoding="utf-8")
    write_figures(report, out / "figures")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="fewer synthetic cases (faster)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("bowlsense", "ultralytics"):
        logging.getLogger(name).setLevel(logging.ERROR)
    report = run(args.quick)
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
