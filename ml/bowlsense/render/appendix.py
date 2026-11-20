"""A live appendix for the PDF report: the user's measurements and notes, and the bowler's
progress across sessions. Merged onto the stored report when it is downloaded."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pypdf import PdfWriter


def _angle(p: Sequence[Sequence[float]]) -> float:
    (ax, ay), (bx, by), (cx, cy) = p[0], p[1], p[2]
    v1, v2 = (ax - bx, ay - by), (cx - bx, cy - by)
    n = math.hypot(*v1) * math.hypot(*v2)
    if n == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))))


def describe_mark(kind: str, points: Sequence[Sequence[float]], label: str) -> str:
    """A measurement in words (angle at the vertex, a line's tilt, or the note text)."""
    if kind == "angle" and len(points) == 3:
        return f"Angle {_angle(points):.1f} deg"
    if kind == "line" and len(points) == 2:
        (ax, ay), (bx, by) = points[0], points[1]
        return f"Line tilted {math.degrees(math.atan2(abs(by - ay), abs(bx - ax))):.1f} deg"
    return f"Note: {label}"


def build_appendix(
    path: Path,
    chart_dir: Path,
    marks: list[tuple[int, dict[str, Any]]],
    progress: list[dict[str, Any]],
    bowler: str | None,
) -> Path | None:
    """Write the appendix PDF; returns None when there is nothing to add."""
    if not marks and len(progress) < 2:
        return None
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table

    styles = getSampleStyleSheet()
    width = A4[0] - 30 * mm
    story: list[Any] = [Paragraph("Appendix", styles["Title"])]
    if marks:
        rows = [["Delivery", "Frame", "Measurement"]]
        for idx, m in marks:
            rows.append([f"#{idx + 1}", str(m["frame"]),
                         describe_mark(m["kind"], m["points"], m.get("label", ""))])  # fmt: skip
        story += [Paragraph("Your measurements and notes", styles["Heading2"]),
                  Table(rows, colWidths=[22 * mm, 18 * mm, width - 40 * mm])]  # fmt: skip
    if len(progress) >= 2:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
        days = [datetime.fromisoformat(str(p["day"])).strftime("%d %b") for p in progress]
        for ax, (key, title) in zip(
            axes,
            [
                ("technique", "Technique /100"),
                ("injury", "Injury load /100"),
                ("elbow_extension_max", "Max elbow straightening (deg)"),
            ],
            strict=True,
        ):
            ax.plot(days, [p[key] for p in progress], marker="o", color="#16140f")
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
            if key == "elbow_extension_max":
                ax.axhline(15, color="#b3261e", linestyle="--", linewidth=1)
        fig.tight_layout()
        png = chart_dir / "progress.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        story += [Spacer(1, 6 * mm),
                  Paragraph(f"Progress{f' of {bowler}' if bowler else ''}", styles["Heading2"]),
                  Image(str(png), width=width, height=width * 2.6 / 9)]  # fmt: skip
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm).build(story)
    return path


def merge_pdfs(parts: list[Path], out: Path) -> Path:
    """Concatenate PDFs into ``out``."""
    writer = PdfWriter()
    for part in parts:
        writer.append(str(part))
    with out.open("wb") as fh:
        writer.write(fh)
    return out
