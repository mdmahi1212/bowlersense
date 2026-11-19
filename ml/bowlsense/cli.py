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
from datetime import datetime
from pathlib import Path

from bowlsense.config import load_settings
from bowlsense.errors import BowlSenseError, ConfigError
from bowlsense.pipeline import analyze_session, extract_pose
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.render.clipping import sub_sequence
from bowlsense.render.keyframes import export_keyframes
from bowlsense.render.overlay import OverlayPainter, OverlaySegment, render_overlay
from bowlsense.render.report_pdf import ReportDelivery, ReportInput, build_report

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
        "--pose-from-json",
        type=Path,
        help="use keypoints from a saved PoseSequence JSON instead of the pose model",
    )
    a.add_argument("--config", type=Path, help="config YAML (default: config/default.yaml)")
    a.add_argument("-v", "--verbose", action="store_true", help="log progress")
    return parser


def analyze(args: argparse.Namespace) -> int:
    """Run the ``analyze`` command."""
    settings = load_settings(args.config) if args.config else load_settings()
    front = extract_pose(
        args.video,
        settings,
        backend=args.backend,
        allow_low_fps=args.allow_low_fps,
        bowler_track_id=args.bowler_track_id,
        pose_json=args.pose_from_json,
    )
    session = analyze_session(front, settings)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "pose.json").write_text(front.sequence.model_dump_json(), encoding="utf-8")

    cleaned = clean_sequence(front.sequence, settings.pose)
    segments = [
        OverlaySegment(start=w.start, stop=w.stop, phases=d.phases, legality=d.legality)
        for w, d in zip(session.windows, session.deliveries, strict=True)
    ]
    overlay = render_overlay(args.video, cleaned, segments, out / "overlay.mp4", settings.render)
    painter = OverlayPainter(cleaned, segments, settings.render)
    report_deliveries = []
    for i, (seg, delivery) in enumerate(zip(segments, session.deliveries, strict=True)):
        keyframes = export_keyframes(
            args.video, painter, seg, out / "keyframes" / f"delivery_{i + 1:02d}", settings.render
        )
        clip_seq = sub_sequence(cleaned, session.windows[i])
        report_deliveries.append(
            ReportDelivery(index=i, analysis=delivery, sequence=clip_seq, keyframes=keyframes)
        )
    report = build_report(
        ReportInput(
            video_name=args.video.name,
            created=datetime.now(),
            view=front.sequence.view,
            bowling_arm=front.sequence.bowling_arm,
            fps=front.video.fps,
            quality_warnings=front.warnings,
            deliveries=report_deliveries,
        ),
        out / "report.pdf",
        out / "charts",
        settings.legality.elbow_extension_threshold_deg,
    )
    result = {
        "video": front.video.model_dump(),
        "backend": front.backend,
        "bowler_track_id": front.bowler_track_id,
        "view": front.sequence.view,
        "bowling_arm": front.sequence.bowling_arm,
        "quality_warnings": front.warnings,
        "deliveries": [
            {"clip": w.model_dump(), **d.model_dump(mode="json")}
            for w, d in zip(session.windows, session.deliveries, strict=True)
        ],
        "fatigue": session.fatigue.model_dump(mode="json"),
        "artifacts": {
            "overlay": overlay.name,
            "report": report.name,
            "pose": "pose.json",
            "keyframes": "keyframes",
        },
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    n = len(session.deliveries)
    print(f"Analysed {n} deliver{'y' if n == 1 else 'ies'}; results in {out}")
    for warning in front.warnings:
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
