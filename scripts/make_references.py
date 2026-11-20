"""Generate the reference actions shipped in data/reference/ from the synthetic generator.

They are textbook *synthetic* actions, named by style, never after a real player.
Run: ml/.venv/Scripts/python scripts/make_references.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from bowlsense.analysis.compare import ReferenceAction
from bowlsense.config import load_settings
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import FramePose, PhaseEvents
from synth.skeleton_gen import SynthParams, generate

REFERENCES = {
    "side_on": (
        "Reference: textbook side-on",
        "Synthetic side-on action: hips and shoulders "
        "closed at back-foot contact, little counter-rotation, firm front leg.",
        {"action_type": "side_on", "counter_rotation_deg": 8.0},
    ),
    "semi_open": (
        "Reference: textbook semi-open",
        "Synthetic semi-open action: hips and " "shoulders aligned between side-on and front-on.",
        {"action_type": "semi_open", "counter_rotation_deg": 8.0},
    ),
    "front_on": (
        "Reference: textbook front-on",
        "Synthetic front-on action: chest facing the "
        "batter at back-foot contact, aligned hips and shoulders.",
        {"action_type": "front_on", "counter_rotation_deg": 6.0},
    ),
}
PAD = 10


def main() -> int:
    """Write one JSON per reference style."""
    settings = load_settings()
    out = settings.compare.reference_path
    out.mkdir(parents=True, exist_ok=True)
    for key, (name, description, params) in REFERENCES.items():
        r = generate(
            SynthParams.model_validate(
                {"elbow_extension_deg": 5.0, "knee_flex_change_deg": 5.0, **params}
            )
        )
        seq = clean_sequence(r.sequence, settings.pose)
        lo, hi = max(0, r.phases.bfc - PAD), min(
            len(seq.frames), r.phases.follow_through_end + PAD + 1
        )
        frames = [
            FramePose(frame_idx=i, t=i / seq.fps, keypoints=f.keypoints)
            for i, f in enumerate(seq.frames[lo:hi])
        ]
        p = r.phases
        phases = PhaseEvents(
            run_up_start=0,
            bound=max(0, p.bound - lo),
            bfc=p.bfc - lo,
            ffc=p.ffc - lo,
            arm_horizontal=p.arm_horizontal - lo,
            release=p.release - lo,
            follow_through_end=p.follow_through_end - lo,
            confidence=1.0,
        )
        ref = ReferenceAction(
            name=name,
            description=description,
            sequence=seq.model_copy(update={"frames": frames}),
            phases=phases,
        )
        path = out / f"{key}.json"
        path.write_text(ref.model_dump_json(), encoding="utf-8")
        print(f"{path.name}: {len(frames)} frames, {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
