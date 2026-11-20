"""Print phase-detection frame errors on synthetic data (dev / evaluation tool).

For each noise condition and bowler size, runs the heuristic detector over several action
variants and seeds and prints the worst signed frame error per event and how many runs
missed by more than 2 frames.

Run: ml/.venv/Scripts/python scripts/phase_error_matrix.py   (Linux: ml/.venv/bin/python)
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from bowlsense.config import load_settings
from bowlsense.geometry.body import estimate_height_px
from bowlsense.phases.heuristic import detect_phases
from bowlsense.pose.smoothing import clean_sequence
from synth.skeleton_gen import SynthParams, generate

EVENTS = (
    "run_up_start",
    "bound",
    "bfc",
    "ffc",
    "arm_horizontal",
    "release",
    "follow_through_end",
)
VARIANTS: list[dict[str, object]] = [
    {},
    {"bowling_arm": "left"},
    {"stride_pct": 95, "knee_flex_change_deg": 30},
    {"action_type": "mixed", "counter_rotation_deg": 35},
    {"runup_decel_pct": 30, "follow_through_pct": 40},
    {"elbow_extension_deg": 22},
    {"release_height_pct": 125},
    {"stride_pct": 62, "action_type": "side_on"},
]
CONDITIONS: list[tuple[str, dict[str, object]]] = [
    ("clean", {}),
    ("noise2", {"noise_px": 2.0}),
    ("drop10", {"dropout_rate": 0.1}),
    ("noise2+drop10", {"noise_px": 2.0, "dropout_rate": 0.1}),
]
SEEDS = range(10)


def main() -> None:
    """Print the error matrix."""
    logging.disable(logging.WARNING)
    s = load_settings()
    for height in (540.0, 300.0):
        for label, cond in CONDITIONS:
            worst = dict.fromkeys(EVENTS, 0)
            over: Counter[str] = Counter()
            runs = 0
            for variant in VARIANTS:
                for seed in SEEDS:
                    p = SynthParams.model_validate(
                        {**variant, **cond, "seed": seed, "height_px": height}
                    )
                    r = generate(p)
                    seq = clean_sequence(r.sequence, s.pose)
                    raw = clean_sequence(r.sequence, s.pose, smooth=False)
                    h = estimate_height_px(seq, s.body).height_px
                    found = detect_phases(seq, h, s.phases, raw=raw)
                    runs += 1
                    for e in EVENTS:
                        err = getattr(found, e) - getattr(r.phases, e)
                        over[e] += abs(err) > 2
                        if abs(err) > abs(worst[e]):
                            worst[e] = err
            misses = {e: c for e, c in over.items() if c}
            print(f"h={height:.0f}px {label:14s} runs={runs} worst={worst} >2 frames={misses}")


if __name__ == "__main__":
    main()
