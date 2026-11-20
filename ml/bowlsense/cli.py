"""Command line: ``python -m bowlsense.cli analyze path/to/video.mp4 --out out/``.

Works with zero web or database setup. Finds every delivery in the clip and writes into
``--out``: ``result.json`` (all deliveries), ``pose.json`` (the bowler's raw keypoints),
``overlay.mp4`` (H.264, plays in a browser), ``keyframes/`` (PNGs per delivery),
``report.pdf`` and ``charts/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bowlsense.config import load_settings
from bowlsense.errors import BowlSenseError, CalibrationError, ConfigError
from bowlsense.geometry.pitch import PitchCalibration, calibrate
from bowlsense.pipeline import PipelineOptions, run_pipeline

log = logging.getLogger("bowlsense.cli")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bowlsense", description="BowlSense bowling analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="analyse a bowling video")
    a.add_argument("video", type=Path, help="video file (mp4/mov/avi)")
    a.add_argument("--out", type=Path, required=True, help="output folder")
    a.add_argument("--allow-low-fps", action="store_true", help="analyse below 60 fps anyway")
    a.add_argument("--backend", choices=["rtm", "mediapipe"], help="pose backend (default: config)")
    a.add_argument("--bowler-track-id", type=int, help="use this tracked person as the bowler")
    a.add_argument(
        "--bowling-arm", choices=["right", "left"], help="the bowler's arm (skips detection)"
    )
    a.add_argument(
        "--pose-from-json",
        type=Path,
        help="use keypoints from a saved PoseSequence JSON instead of the pose model",
    )
    a.add_argument(
        "--calib",
        type=Path,
        help='pitch calibration JSON: {"points": [[x, y] x 4]} (crease corners) for ball '
        "speed, length and line",
    )
    a.add_argument(
        "--slow-motion",
        type=float,
        help="how many times the clip is slowed down (1 = real time; default: detect it)",
    )
    a.add_argument(
        "--view",
        choices=["side", "front", "behind"],
        help="camera view (default: estimate it)",
    )
    a.add_argument(
        "--no-llm", action="store_true", help="skip the local Ollama coach; use the template report"
    )
    a.add_argument("--config", type=Path, help="config YAML (default: config/default.yaml)")
    a.add_argument("-v", "--verbose", action="store_true", help="log progress")
    return parser


def load_calibration(path: Path) -> PitchCalibration:
    """Read ``{"points": [[x, y], ...]}`` and solve the pitch calibration."""
    if not path.is_file():
        raise CalibrationError(f"Calibration file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        points = data["points"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CalibrationError(
            f'{path.name} must be JSON like {{"points": [[x, y], ...]}}'
        ) from exc
    return calibrate(points)


def analyze(args: argparse.Namespace) -> int:
    """Run the ``analyze`` command (no web app or database needed)."""
    settings = load_settings(args.config) if args.config else load_settings()
    calib = load_calibration(args.calib) if args.calib else None
    points = [list(p) for p in calib.image_points] if calib else None
    options = PipelineOptions(
        backend=args.backend,
        allow_low_fps=args.allow_low_fps,
        bowler_track_id=args.bowler_track_id,
        pose_json=args.pose_from_json,
        bowling_arm=args.bowling_arm,
        calibration_points=points,
        use_llm=not args.no_llm,
        slow_motion=args.slow_motion,
        camera_view=args.view,
    )
    out: Path = args.out
    result = run_pipeline(args.video, out, settings, session_id=out.name, options=options)
    n = len(result.deliveries)
    print(f"Analysed {n} deliver{'y' if n == 1 else 'ies'}; results in {out}")
    if result.coach_note:
        print(f"coach: {result.coach_note}", file=sys.stderr)
    for warning in result.quality_warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code: 0 ok, 2 for an input/analysis problem."""
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "analyze":
            return analyze(args)
    except (BowlSenseError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1  # pragma: no cover - argparse rejects unknown commands


if __name__ == "__main__":
    sys.exit(main())
