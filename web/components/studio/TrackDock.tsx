"use client";

import { useRef } from "react";
import type { PhaseEvents } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface Lane {
  label: string;
  unit: string;
  values: (number | null)[]; // one per clip frame
  colour: string;
}

export interface Marker {
  frame: number;
  label: string;
}

const PHASES: [keyof PhaseEvents, keyof PhaseEvents, string, string][] = [
  ["run_up_start", "bound", "Run-up", "#a39a87"],
  ["bound", "bfc", "Bound", "#8f8672"],
  ["bfc", "ffc", "Delivery stride", "#7a7262"],
  ["ffc", "arm_horizontal", "Brace", "#5f584b"],
  ["arm_horizontal", "release", "Arm action", "#b3261e"],
  ["release", "follow_through_end", "Follow-through", "#3f3a31"],
];

/** Video-editor style timeline: phase lane, data lanes with sparklines, fault markers,
 *  playhead and release line. Click or drag anywhere to scrub. */
export function TrackDock({
  length,
  frame,
  phases,
  lanes,
  markers,
  onSeek,
  fps,
}: {
  length: number;
  frame: number;
  phases: PhaseEvents;
  lanes: Lane[];
  markers: Marker[];
  onSeek: (f: number) => void;
  fps: number;
}) {
  const area = useRef<HTMLDivElement>(null);
  const last = Math.max(1, length - 1);
  const x = (f: number) => `${(Math.min(last, Math.max(0, f)) / last) * 100}%`;
  const seekAt = (clientX: number) => {
    const r = area.current?.getBoundingClientRect();
    if (r) onSeek(((clientX - r.left) / r.width) * last);
  };
  const ticks = Array.from({ length: 9 }, (_, i) => Math.round((i / 8) * last));

  return (
    <div className="grid grid-cols-[8.5rem_1fr] overflow-hidden border border-border bg-surface text-[11px]">
      <div className="border-r border-border bg-surface-2/60">
        <div className="h-6 border-b border-border px-2 leading-6 text-muted">Time</div>
        <div className="h-7 border-b border-border px-2 leading-7 font-display uppercase tracking-wider">
          Phases
        </div>
        {lanes.map((l) => (
          <div
            key={l.label}
            className="flex h-12 flex-col justify-center border-b border-border px-2"
          >
            <span className="font-display uppercase tracking-wider" style={{ color: l.colour }}>
              {l.label}
            </span>
            <span className="tabular text-sm text-foreground">
              {fmtValue(l.values[frame])}
              <span className="ml-1 text-[10px] text-muted">{l.unit}</span>
            </span>
          </div>
        ))}
        <div className="h-7 px-2 leading-7 font-display uppercase tracking-wider">Events</div>
      </div>
      <div
        ref={area}
        className="relative cursor-crosshair select-none"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          seekAt(e.clientX);
        }}
        onPointerMove={(e) => e.buttons === 1 && seekAt(e.clientX)}
        role="slider"
        aria-label="Timeline"
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={frame}
      >
        <div className="relative h-6 border-b border-border">
          {ticks.map((t) => (
            <span
              key={t}
              className="tabular absolute top-1 -translate-x-1/2 text-muted"
              style={{ left: x(t) }}
            >
              {(((t - phases.release) / fps) * 1000).toFixed(0)}ms
            </span>
          ))}
        </div>
        <div className="relative h-7 border-b border-border">
          {PHASES.map(([a, b, label, colour]) => (
            <div
              key={label}
              className="absolute inset-y-1 overflow-hidden px-1 leading-5 text-[10px] text-white/85"
              style={{
                left: x(phases[a]),
                width: `calc(${x(phases[b])} - ${x(phases[a])})`,
                background: colour,
              }}
              title={label}
            >
              {label}
            </div>
          ))}
        </div>
        {lanes.map((l) => (
          <Spark key={l.label} lane={l} last={last} />
        ))}
        <div className="relative h-7">
          {markers.map((m, i) => (
            <div
              key={`${m.label}${i}`}
              className="absolute top-1.5 -translate-x-1/2"
              style={{ left: x(m.frame) }}
              title={m.label}
            >
              <div className="h-3 w-3 rotate-45 border border-bad bg-bad/40" />
            </div>
          ))}
        </div>
        <div
          className="pointer-events-none absolute inset-y-0 w-0.5 bg-release/90"
          style={{ left: x(phases.release) }}
        >
          <span className="absolute -top-0 left-1 bg-release px-1 text-[9px] font-bold text-white">
            REL
          </span>
        </div>
        <div
          className="pointer-events-none absolute inset-y-0 w-px bg-accent shadow-[0_0_6px_var(--accent)]"
          style={{ left: x(frame) }}
        >
          <div className="absolute -left-1 top-0 h-2 w-2.5 bg-accent [clip-path:polygon(0_0,100%_0,50%_100%)]" />
        </div>
      </div>
    </div>
  );
}

function fmtValue(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1);
}

function Spark({ lane, last }: { lane: Lane; last: number }) {
  const vals = lane.values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (vals.length < 2)
    return (
      <div className="h-12 border-b border-border px-2 leading-[3rem] text-muted">
        not measurable from this view
      </div>
    );
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = hi - lo || 1;
  let d = "";
  let pen = false;
  lane.values.forEach((v, i) => {
    if (v === null || !Number.isFinite(v)) {
      pen = false;
      return;
    }
    const px = (i / last) * 1000;
    const py = 44 - ((v - lo) / span) * 38;
    d += `${pen ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`;
    pen = true;
  });
  return (
    <div className={cn("relative h-12 border-b border-border")}>
      <svg
        viewBox="0 0 1000 48"
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full"
        aria-hidden="true"
      >
        <path
          d={`${d}`}
          fill="none"
          stroke={lane.colour}
          strokeWidth={1.6}
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <span className="tabular absolute right-1 top-0.5 text-[9px] text-muted">{fmtValue(hi)}</span>
      <span className="tabular absolute bottom-0.5 right-1 text-[9px] text-muted">
        {fmtValue(lo)}
      </span>
    </div>
  );
}
