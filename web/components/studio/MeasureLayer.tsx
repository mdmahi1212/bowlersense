"use client";

import { useState, type MouseEvent } from "react";
import type { Annotation } from "@/lib/types";

export type MeasureMode = "off" | "line" | "angle" | "note";
type Pt = [number, number];

/** Angle at ``b`` between ``a`` and ``c``, in degrees (0-180). */
export function angleAt(a: Pt, b: Pt, c: Pt): number {
  const v1 = [a[0] - b[0], a[1] - b[1]];
  const v2 = [c[0] - b[0], c[1] - b[1]];
  const n = Math.hypot(v1[0], v1[1]) * Math.hypot(v2[0], v2[1]);
  if (n === 0) return 0;
  const cos = Math.max(-1, Math.min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / n));
  return (Math.acos(cos) * 180) / Math.PI;
}

/** A line's tilt from horizontal, in degrees (0-90). */
export function tilt(a: Pt, b: Pt): number {
  const d = (Math.atan2(Math.abs(b[1] - a[1]), Math.abs(b[0] - a[0])) * 180) / Math.PI;
  return d;
}

export function describe(a: Annotation): string {
  const p = a.points as Pt[];
  if (a.kind === "note") return `“${a.label}”`;
  return a.kind === "angle" && p.length === 3
    ? `${angleAt(p[0], p[1], p[2]).toFixed(1)}°`
    : `${tilt(p[0], p[1]).toFixed(1)}° from horizontal`;
}

/** Lines and angles drawn on the paused frame, in video-pixel coordinates (an SVG laid over
 *  the object-contain video with the same aspect ratio, so marks stay on the body). */
export function MeasureLayer({
  width,
  height,
  frame,
  annotations,
  mode,
  onAdd,
}: {
  width: number;
  height: number;
  frame: number;
  annotations: Annotation[];
  mode: MeasureMode;
  onAdd: (a: Annotation) => void;
}) {
  const [draft, setDraft] = useState<Pt[]>([]);
  const [hover, setHover] = useState<Pt | null>(null);
  const need = mode === "angle" ? 3 : mode === "note" ? 1 : 2;
  const u = Math.max(width, height) / 400; // stroke / text unit

  function toVideo(e: MouseEvent<SVGSVGElement>): Pt | null {
    const svg = e.currentTarget;
    const m = svg.getScreenCTM();
    if (!m) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    return [Math.round(p.x), Math.round(p.y)];
  }
  function click(e: MouseEvent<SVGSVGElement>) {
    if (mode === "off") return;
    e.stopPropagation();
    const p = toVideo(e);
    if (!p) return;
    const pts = [...draft, p];
    if (mode === "note") {
      const text = window.prompt("Coach note for this frame:")?.trim();
      if (text) onAdd({ kind: "note", frame, points: [p], label: text.slice(0, 500) });
      return;
    }
    if (pts.length >= need) {
      onAdd({ kind: mode, frame, points: pts as Annotation["points"], label: "" });
      setDraft([]);
    } else setDraft(pts);
  }

  const shown = annotations.filter((a) => a.frame === frame);
  const live = mode !== "off" && draft.length > 0 && hover ? [...draft, hover] : draft;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      className="absolute inset-0 h-full w-full"
      style={{
        pointerEvents: mode === "off" ? "none" : "auto",
        cursor: mode === "off" ? undefined : "crosshair",
      }}
      onClick={click}
      onMouseMove={(e) => mode !== "off" && setHover(toVideo(e))}
      onMouseLeave={() => setHover(null)}
      onContextMenu={(e) => {
        e.preventDefault();
        setDraft([]);
      }}
      aria-hidden={mode === "off"}
    >
      {shown.map((a, i) =>
        a.kind === "note" ? (
          <Note key={i} at={a.points[0] as Pt} text={a.label} u={u} />
        ) : (
          <Shape key={i} pts={a.points as Pt[]} kind={a.kind} u={u} />
        ),
      )}
      {live.length > 0 && (
        <Shape pts={live} kind={live.length === 3 ? "angle" : "line"} u={u} draft />
      )}
    </svg>
  );
}

function Shape({
  pts,
  kind,
  u,
  draft = false,
}: {
  pts: Pt[];
  kind: "line" | "angle";
  u: number;
  draft?: boolean;
}) {
  const colour = draft ? "#ffc94a" : "#ffd84d";
  const text = (x: number, y: number, s: string) => (
    <g>
      <text
        x={x}
        y={y}
        fontSize={11 * u}
        fontFamily="var(--font-mono)"
        stroke="#000"
        strokeWidth={3 * u}
        paintOrder="stroke"
        fill={colour}
      >
        {s}
      </text>
    </g>
  );
  let label: React.ReactNode = null;
  let arc: React.ReactNode = null;
  if (kind === "angle" && pts.length === 3) {
    const [a, b, c] = pts;
    const deg = angleAt(a, b, c);
    const r = 22 * u;
    const a1 = Math.atan2(a[1] - b[1], a[0] - b[0]);
    const a2 = Math.atan2(c[1] - b[1], c[0] - b[0]);
    let d = a2 - a1;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    const p1 = [b[0] + r * Math.cos(a1), b[1] + r * Math.sin(a1)];
    const p2 = [b[0] + r * Math.cos(a1 + d), b[1] + r * Math.sin(a1 + d)];
    arc = (
      <path
        d={`M ${p1[0]} ${p1[1]} A ${r} ${r} 0 0 ${d > 0 ? 1 : 0} ${p2[0]} ${p2[1]}`}
        fill="none"
        stroke={colour}
        strokeWidth={1.5 * u}
      />
    );
    label = text(b[0] + 26 * u, b[1] - 8 * u, `${deg.toFixed(1)}°`);
  } else if (pts.length >= 2) {
    const [a, b] = pts;
    label = text((a[0] + b[0]) / 2 + 8 * u, (a[1] + b[1]) / 2 - 8 * u, `${tilt(a, b).toFixed(1)}°`);
  }
  return (
    <g>
      <polyline
        points={pts.map((p) => p.join(",")).join(" ")}
        fill="none"
        stroke={colour}
        strokeWidth={2 * u}
        strokeDasharray={draft ? `${4 * u} ${3 * u}` : undefined}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {pts.map((p, i) => (
        <circle
          key={i}
          cx={p[0]}
          cy={p[1]}
          r={3.5 * u}
          fill="#000"
          stroke={colour}
          strokeWidth={1.5 * u}
        />
      ))}
      {arc}
      {label}
    </g>
  );
}

/** A coach's note pinned to a point on the frame. */
function Note({ at, text, u }: { at: Pt; text: string; u: number }) {
  const lines = text.match(/.{1,34}(\s|$)/g)?.map((l) => l.trim()) ?? [text];
  const w = Math.min(34, Math.max(...lines.map((l) => l.length))) * 6.2 * u + 16 * u;
  const h = lines.length * 13 * u + 12 * u;
  return (
    <g>
      <circle cx={at[0]} cy={at[1]} r={5 * u} fill="#ffd84d" stroke="#000" strokeWidth={u} />
      <line
        x1={at[0]}
        y1={at[1]}
        x2={at[0] + 18 * u}
        y2={at[1] - 18 * u}
        stroke="#ffd84d"
        strokeWidth={1.5 * u}
      />
      <rect
        x={at[0] + 18 * u}
        y={at[1] - 18 * u - h}
        width={w}
        height={h}
        fill="#16140f"
        fillOpacity={0.88}
        stroke="#ffd84d"
        strokeWidth={u}
      />
      {lines.map((l, i) => (
        <text
          key={i}
          x={at[0] + 26 * u}
          y={at[1] - 18 * u - h + 16 * u + i * 13 * u}
          fontSize={11 * u}
          fill="#f2eee5"
        >
          {l}
        </text>
      ))}
    </g>
  );
}
