"use client";

import type { Arm } from "@/lib/pose";
import { jointName } from "@/lib/pose";
import type { Fault, FramePose, PhaseEvents } from "@/lib/types";

const EVENTS: (keyof PhaseEvents)[] = [
  "run_up_start",
  "bound",
  "bfc",
  "ffc",
  "arm_horizontal",
  "release",
  "follow_through_end",
];

/** The joints each fault is about (for highlighting on the video and the figures). */
export function faultJoints(code: string, arm: Arm): string[] {
  const b = (p: string) => jointName(arm, "bowling", p);
  const fr = (p: string) => jointName(arm, "front", p);
  switch (code) {
    case "KNEE_COLLAPSE":
      return [fr("knee")];
    case "OVERSTRIDE":
    case "UNDERSTRIDE":
      return ["l_ankle", "r_ankle"];
    case "HEAD_FALL":
      return ["nose", fr("ankle")];
    case "LATERAL_FLEXION":
      return ["l_shoulder", "r_shoulder", "l_hip", "r_hip"];
    case "WEAK_FRONT_ARM":
      return [fr("wrist"), fr("shoulder")];
    case "RUNUP_DECEL":
    case "SHORT_FOLLOW":
      return ["l_hip", "r_hip"];
    case "LOW_RELEASE":
      return [b("wrist")];
    case "CHAIN_ORDER":
      return ["l_hip", "r_hip", b("shoulder"), b("elbow")];
    default:
      return [];
  }
}

/** The reference frame at the same point of the action as ``frame`` (phase-relative). */
export function matchingFrame(frame: number, mine: PhaseEvents, ref: PhaseEvents): number {
  for (let i = 0; i < EVENTS.length - 1; i += 1) {
    const a = mine[EVENTS[i]] as number;
    const b = mine[EVENTS[i + 1]] as number;
    if (frame >= a && frame <= b && b > a) {
      const u = (frame - a) / (b - a);
      const ra = ref[EVENTS[i]] as number;
      const rb = ref[EVENTS[i + 1]] as number;
      return Math.round(ra + u * (rb - ra));
    }
  }
  return frame < (mine.run_up_start as number)
    ? (ref.run_up_start as number)
    : (ref.follow_through_end as number);
}

function Figure({
  f,
  edges,
  hot,
  colour,
}: {
  f: FramePose | undefined;
  edges: string[][];
  hot: string[];
  colour: string;
}) {
  const ks = Object.values(f?.keypoints ?? {}).filter(Boolean) as { x: number; y: number }[];
  if (ks.length < 4)
    return <div className="grid h-40 place-items-center text-xs text-muted">no skeleton</div>;
  const x0 = Math.min(...ks.map((k) => k.x));
  const x1 = Math.max(...ks.map((k) => k.x));
  const y0 = Math.min(...ks.map((k) => k.y));
  const y1 = Math.max(...ks.map((k) => k.y));
  const pad = 0.12 * Math.max(x1 - x0, y1 - y0);
  const vb = `${x0 - pad} ${y0 - pad} ${x1 - x0 + 2 * pad} ${y1 - y0 + 2 * pad}`;
  const s = Math.max(x1 - x0, y1 - y0) / 100;
  const k = f!.keypoints;
  return (
    <svg viewBox={vb} className="h-40 w-full">
      {edges.map(([a, b]) =>
        k[a] && k[b] ? (
          <line
            key={a + b}
            x1={k[a]!.x}
            y1={k[a]!.y}
            x2={k[b]!.x}
            y2={k[b]!.y}
            stroke={hot.includes(a) || hot.includes(b) ? "var(--accent)" : colour}
            strokeWidth={2.2 * s}
            strokeLinecap="round"
          />
        ) : null,
      )}
      {hot.map((n) =>
        k[n] ? (
          <circle
            key={n}
            cx={k[n]!.x}
            cy={k[n]!.y}
            r={5 * s}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={1.6 * s}
          />
        ) : null,
      )}
    </svg>
  );
}

/** "Fix this fault": the bowler's skeleton at the fault next to the reference action at the
 *  same moment of the delivery, with the joints that matter highlighted. */
export function FaultFocus({
  faults,
  frames,
  clipStart,
  phases,
  arm,
  edges,
  reference,
  onShow,
}: {
  faults: Fault[];
  frames: Map<number, FramePose> | null;
  clipStart: number;
  phases: PhaseEvents;
  arm: Arm;
  edges: string[][];
  reference: { name: string; frames: Map<number, FramePose>; phases: PhaseEvents } | null;
  onShow: (frame: number, code: string) => void;
}) {
  if (faults.length === 0 || !frames) return null;
  return (
    <div className="grid gap-6 md:grid-cols-2">
      {faults.map((f, i) => {
        const at = f.frames[0] ?? phases.release;
        const hot = faultJoints(f.code, arm);
        const refF = reference
          ? reference.frames.get(matchingFrame(at, phases, reference.phases))
          : undefined;
        return (
          <figure
            key={f.code}
            data-reveal
            style={{ "--d": `${i * 80}ms` } as React.CSSProperties}
            className="border-t border-foreground pt-3"
          >
            <figcaption className="flex items-baseline justify-between gap-3">
              <span className="font-display text-xl">{f.name}</span>
              <button
                type="button"
                onClick={() => onShow(at, f.code)}
                className="shrink-0 text-sm text-accent hover:underline"
              >
                Show on video →
              </button>
            </figcaption>
            <div className="mt-3 grid grid-cols-2 gap-3">
              <div>
                <Figure
                  f={frames.get(clipStart + at)}
                  edges={edges}
                  hot={hot}
                  colour="var(--foreground)"
                />
                <p className="kicker mt-1 text-center">You · frame {at}</p>
              </div>
              <div>
                <Figure f={refF} edges={edges} hot={hot} colour="var(--muted)" />
                <p className="kicker mt-1 text-center">
                  {reference ? "Reference" : "No reference"}
                </p>
              </div>
            </div>
            <p className="mt-2 text-sm text-muted">
              Measured <span className="tabular text-foreground">{f.value.toFixed(1)}</span> against
              a limit of <span className="tabular">{f.threshold.toFixed(1)}</span>.
            </p>
          </figure>
        );
      })}
    </div>
  );
}
