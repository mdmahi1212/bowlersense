"""PDF report (ReportLab) with matplotlib charts.

Page 1: summary (technique score, injury band, legality, action type) and the disclaimers.
Per delivery: fault table, keyframe strip, angle charts, injury breakdown. A coach section is
included when coach text is given. Every page footer repeats the disclaimers.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from bowlsense.analysis.legality import elbow_angles
from bowlsense.geometry.angles import FloatArray, keypoint_track, midpoint
from bowlsense.render.overlay import EVENT_LABELS, front_knee_angles
from bowlsense.schemas import DeliveryAnalysis, InjuryRisk, PoseSequence

LEGALITY_NOTE = "2D video screening only. The official ICC test uses 3D motion capture."
INJURY_NOTE = (
    "Injury risk is a rule-based screening estimate, not a trained medical model and not "
    "medical advice. See a qualified professional about pain or injury."
)


class ReportDelivery(BaseModel):
    """One delivery's content for the report."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    analysis: DeliveryAnalysis
    sequence: PoseSequence  # cleaned keypoints, for the charts
    keyframes: dict[str, Path]


class ReportInput(BaseModel):
    """Everything the PDF needs."""

    title: str = "BowlSense bowling report"
    video_name: str
    created: datetime
    view: str
    bowling_arm: str
    fps: float
    quality_warnings: list[str]
    deliveries: list[ReportDelivery]
    coach_text: str | None = None


def trunk_lean(seq: PoseSequence) -> FloatArray:
    """Trunk angle from vertical (deg) in the image: hip midpoint -> shoulder midpoint."""
    sh = midpoint(keypoint_track(seq, "l_shoulder"), keypoint_track(seq, "r_shoulder"))
    hp = midpoint(keypoint_track(seq, "l_hip"), keypoint_track(seq, "r_hip"))
    dx, dy_up = sh[:, 0] - hp[:, 0], hp[:, 1] - sh[:, 1]
    return np.asarray(np.degrees(np.arctan2(np.abs(dx), dy_up)), dtype=np.float64)


def angle_chart(delivery: ReportDelivery, path: Path, threshold_deg: float) -> Path:
    """Elbow (with the legality band), front-knee and trunk-lean angles over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq, ph = delivery.sequence, delivery.analysis.phases
    t = np.array([f.t for f in seq.frames]) - ph.release / seq.fps
    series = [
        ("Bowling elbow (deg)", elbow_angles(seq)),
        ("Front knee (deg)", front_knee_angles(seq)),
        ("Trunk lean from vertical (deg)", trunk_lean(seq)),
    ]
    lo = max(0, ph.bfc - round(0.4 * seq.fps))
    hi = min(len(seq.frames) - 1, ph.follow_through_end)
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.2), sharex=True)
    try:
        for ax, (label, values) in zip(axes, series, strict=True):
            ax.plot(t[lo : hi + 1], values[lo : hi + 1], color="#1f6feb", linewidth=1.6)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(alpha=0.3)
            for event in ("bfc", "ffc", "arm_horizontal", "release"):
                x = getattr(ph, event) / seq.fps - ph.release / seq.fps
                ax.axvline(x, color="#888888", linewidth=0.8, linestyle="--")
            ax.tick_params(labelsize=7)
        window = values_in(series[0][1], ph.arm_horizontal, ph.release)
        if window.size:
            base = float(np.nanmin(window))
            axes[0].axhspan(base, base + threshold_deg, color="#f0a500", alpha=0.18,
                            label=f"{threshold_deg:.0f} deg extension limit")  # fmt: skip
            axes[0].legend(fontsize=7, loc="lower left")
        top = axes[0].get_ylim()[1]
        for event in ("bfc", "ffc", "release"):
            x = getattr(ph, event) / seq.fps - ph.release / seq.fps
            axes[0].text(x, top, EVENT_LABELS[event], fontsize=6, rotation=90, va="top",
                         ha="right", color="#555555")  # fmt: skip
        axes[-1].set_xlabel("seconds from release", fontsize=8)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return path


def values_in(values: FloatArray, lo: int, hi: int) -> FloatArray:
    """Finite values of ``values[lo:hi+1]``."""
    seg = values[lo : hi + 1]
    return seg[np.isfinite(seg)]


def _ball_text(a: DeliveryAnalysis) -> str:
    """Speed / length / line, or why they're missing."""
    if a.ball is None:
        return "not tracked"
    parts = []
    if a.ball.speed_kmh is not None:
        parts.append(f"{a.ball.speed_kmh:.0f} km/h")
    parts += [p for p in (a.ball.length, a.ball.line) if p]
    if a.delivery_type is not None:
        parts.append(a.delivery_type.delivery_type)
    return ", ".join(parts) if parts else "; ".join(a.ball.notes) or "n/a"


def _injury_rows(risk: InjuryRisk) -> list[list[str]]:
    rows = [["Component", "Score", "Weight", "Drivers"]]
    for c in risk.components:
        rows.append(
            [c.name.replace("_", " "), f"{c.score:.0f}", f"{c.weight:.2f}", "; ".join(c.drivers)]
        )
    return rows


def build_report(data: ReportInput, path: Path, chart_dir: Path, threshold_deg: float) -> Path:
    """Write the PDF and return its path."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=10)
    note = ParagraphStyle("note", parent=small, textColor=colors.HexColor("#7a4b00"))
    width = A4[0] - 30 * mm

    def para(text: str, style: Any = small) -> Paragraph:
        return Paragraph(text.replace("&", "&amp;").replace("<", "&lt;"), style)

    def table(rows: list[list[Any]], col_widths: list[float] | None = None) -> Table:
        wrapped = [[c if not isinstance(c, str) else para(c) for c in row] for row in rows]
        t = Table(wrapped, colWidths=col_widths, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b8c4")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        return t

    story: list[Any] = [
        Paragraph(data.title, styles["Title"]),
        para(
            f"Video: {data.video_name} &middot; {data.created:%Y-%m-%d %H:%M} &middot; "
            f"{data.view} view &middot; {data.bowling_arm}-arm bowler &middot; {data.fps:.0f} fps"
        ),
        Spacer(1, 4 * mm),
    ]
    summary = [["Delivery", "Technique score", "Injury risk", "Legality (elbow)", "Action", "Ball"]]
    for d in data.deliveries:
        a = d.analysis
        leg = a.legality
        summary.append(
            [
                f"#{d.index + 1}",
                f"{a.faults.technique_score:.0f} / 100",
                f"{a.injury.overall:.0f} ({a.injury.band})",
                (
                    f"{'NOT LEGAL' if leg.flagged else 'LEGAL'}: "
                    f"{leg.elbow_extension_deg:.1f} deg (limit {leg.threshold_deg:.0f}; "
                    f"{leg.confidence} confidence)"
                    if leg.screened
                    else "CAN'T JUDGE (see note)"
                ),
                a.alignment.action_type.replace("_", " "),
                _ball_text(a),
            ]
        )
    story += [Paragraph("Summary", styles["Heading2"]), table(summary)]
    box = Table([[para(f"<b>Legality:</b> {LEGALITY_NOTE}", note)], [para(INJURY_NOTE, note)]],
                colWidths=[width])  # fmt: skip
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#f0a500")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7e6")),
            ]
        )
    )
    story += [Spacer(1, 3 * mm), box]
    if data.quality_warnings:
        story += [Paragraph("Recording quality", styles["Heading3"])]
        story += [para(f"&bull; {w}") for w in data.quality_warnings]

    for d in data.deliveries:
        a = d.analysis
        story += [PageBreak(), Paragraph(f"Delivery #{d.index + 1}", styles["Heading2"])]
        fault_rows: list[list[Any]] = [["Fault", "Value", "Limit", "Severity", "What it means"]]
        for f in a.faults.faults:
            fault_rows.append(
                [f.name, f"{f.value:.1f}", f"{f.threshold:.1f}", f"{f.severity:.0f}", f.explanation]
            )
        if len(fault_rows) == 1:
            fault_rows.append(
                ["No faults", "", "", "", "Every measurable check is within its limit."]
            )
        fault_cols = [32 * mm, 16 * mm, 14 * mm, 16 * mm, width - 78 * mm]
        story += [Paragraph("Technique faults", styles["Heading3"]), table(fault_rows, fault_cols)]
        if a.faults.not_measured:
            missing = ", ".join(f"{k} ({v})" for k, v in a.faults.not_measured.items())
            story.append(para(f"Not measured from this view: {missing}", note))

        if d.keyframes:
            cell = width / len(d.keyframes)
            images, labels = [], []
            for event, img_path in d.keyframes.items():
                image = Image(str(img_path))
                scale = min(cell / image.imageWidth, 45 * mm / image.imageHeight)
                image.drawWidth, image.drawHeight = (
                    image.imageWidth * scale,
                    image.imageHeight * scale,
                )
                images.append(image)
                labels.append(para(EVENT_LABELS[event]))
            strip = Table([images, labels], colWidths=[cell] * len(images))
            story += [Paragraph("Key moments", styles["Heading3"]), strip]

        chart = angle_chart(d, chart_dir / f"delivery_{d.index + 1:02d}_angles.png", threshold_deg)
        img = Image(str(chart), width=width, height=width * 6.2 / 7.0)
        story += [KeepTogether([Paragraph("Joint angles", styles["Heading3"]), img])]
        injury_cols = [24 * mm, 14 * mm, 14 * mm, width - 52 * mm]
        story += [
            Paragraph("Injury-risk screening", styles["Heading3"]),
            table(_injury_rows(a.injury), injury_cols),
        ]
        for n in a.injury.notes:
            story.append(para(n, note))

    if data.coach_text:
        story += [PageBreak(), Paragraph("Coach's notes", styles["Heading2"])]
        story += [para(p) for p in data.coach_text.split("\n") if p.strip()]

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(15 * mm, 8 * mm, f"{LEGALITY_NOTE} Injury risk is not medical advice.")
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"BowlSense - page {doc.page}")
        canvas.restoreState()

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=14 * mm, bottomMargin=16 * mm, title=data.title,
                            author="BowlSense")  # fmt: skip
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path
