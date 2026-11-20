"use client";

import { useEffect, useRef } from "react";
import { type Arm, type FrameAngles, fmtDeg, frameAngles, jointName } from "@/lib/pose";
import type { BiomechMetric, DeliveryMetrics, FramePose } from "@/lib/types";
import { PHASE_EVENTS, cn } from "@/lib/utils";

const ROWS: [keyof FrameAngles, string, string][] = [
  ["elbow", "Bowling elbow", "180° = straight"],
  ["frontKnee", "Front knee", "braced ≈ 160°+"],
  ["backKnee", "Back knee", ""],
  ["trunkLean", "Trunk lean", "from vertical"],
  ["upperArm", "Upper arm", "above horizontal"],
  ["shoulderAbduction", "Shoulder abduction", "arm to trunk"],
];

/** Angles at the current frame (live while scrubbing). */
export function LiveAngles({
  pose,
  arm,
  phase,
}: {
  pose: FramePose | undefined;
  arm: Arm;
  phase: string;
}) {
  const a = frameAngles(pose, arm);
  return (
    <div>
      <p className="mb-2 font-display text-xs uppercase tracking-widest text-accent">{phase}</p>
      <dl className="space-y-2">
        {ROWS.map(([key, label, hint]) => {
          const v = a[key];
          return (
            <div key={key}>
              <div className="flex items-baseline justify-between text-sm">
                <dt>
                  {label} {hint && <span className="text-xs text-muted">· {hint}</span>}
                </dt>
                <dd className="tabular text-base font-semibold">{fmtDeg(v)}</dd>
              </div>
              <div className="mt-1 h-1 rounded-full bg-surface-2">
                <div
                  className="h-1 rounded-full bg-accent"
                  style={{ width: `${v === null ? 0 : Math.min(100, (v / 180) * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </dl>
    </div>
  );
}

/** A frame of the video at ``frame`` with the skeleton drawn on top (no controls). */
export function FrameStill({
  src,
  fps,
  videoFrame,
  pose,
  arm,
  edges,
  className,
}: {
  src: string;
  fps: number;
  videoFrame: number;
  pose: FramePose | undefined;
  arm: Arm;
  edges: string[][];
  className?: string;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const v = video.current;
    if (!v) return;
    const seek = () => {
      v.currentTime = (videoFrame + 0.5) / fps;
    };
    if (v.readyState >= 1) seek();
    else v.addEventListener("loadedmetadata", seek, { once: true });
  }, [videoFrame, fps]);
  function paint() {
    const v = video.current;
    const c = canvas.current;
    if (!v || !c || !pose) return;
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const s = Math.max(1.5, c.width / 500);
    const bowl = jointName(arm, "bowling", "");
    const front = jointName(arm, "front", "");
    for (const [a, b] of edges) {
      const p = pose.keypoints[a];
      const q = pose.keypoints[b];
      if (!p || !q) continue;
      ctx.strokeStyle =
        a.startsWith(bowl) && b.startsWith(bowl) && /shoulder|elbow|wrist/.test(a + b)
          ? "#c6f432"
          : a.startsWith(front) && b.startsWith(front) && /hip|knee|ankle/.test(a + b)
            ? "#ff7a2f"
            : "rgba(255,255,255,0.8)";
      ctx.lineWidth = 3 * s;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(q.x, q.y);
      ctx.stroke();
    }
  }
  return (
    <div className={cn("relative overflow-hidden rounded-sm bg-black", className)}>
      <video
        ref={video}
        src={src}
        muted
        playsInline
        preload="auto"
        className="block w-full"
        onSeeked={paint}
      />
      <canvas ref={canvas} className="pointer-events-none absolute inset-0 h-full w-full" />
    </div>
  );
}

/** The release frame: still, verdict and the angles that matter at that instant. */
export function ReleaseSpotlight({
  d,
  pose,
  arm,
  edges,
  src,
  fps,
  onOpen,
}: {
  d: DeliveryMetrics;
  pose: FramePose | undefined;
  arm: Arm;
  edges: string[][];
  src: string;
  fps: number;
  onOpen: () => void;
}) {
  const a = frameAngles(pose, arm);
  const bm = new Map(d.biomechanics.map((m) => [m.key, m]));
  const tiles: [string, string, string?][] = [
    ["Elbow at release", fmtDeg(a.elbow)],
    [
      "Elbow straightening",
      d.legality.screened ? fmtDeg(d.legality.elbow_extension_deg, 1) : "—",
      "limit 15°",
    ],
    ["Front knee", fmtDeg(a.frontKnee)],
    ["Trunk lean", fmtDeg(a.trunkLean)],
    ["Upper arm", fmtDeg(a.upperArm)],
    ["Release height", metric(bm.get("release_height")), "% of height"],
  ];
  return (
    <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
      <button
        type="button"
        onClick={onOpen}
        className="group relative text-left"
        aria-label="Jump the player to the release frame"
      >
        <FrameStill
          src={src}
          fps={fps}
          videoFrame={d.clip_start + d.phases.release}
          pose={pose}
          arm={arm}
          edges={edges}
          className="border border-release"
        />
        <span className="absolute left-2 top-2 rounded-sm bg-release px-2 py-0.5 font-display text-xs font-bold uppercase tracking-widest text-black">
          Release · frame {d.phases.release}
        </span>
        <span className="absolute bottom-2 right-2 rounded-sm bg-black/70 px-2 py-0.5 text-xs opacity-0 transition group-hover:opacity-100">
          Open in player
        </span>
      </button>
      <div className="grid grid-cols-2 gap-2 content-start">
        {tiles.map(([label, value, hint]) => (
          <div key={label} className="rounded-sm border border-border bg-surface-2 p-3">
            <p className="font-display text-[11px] uppercase tracking-widest text-muted">{label}</p>
            <p className="tabular mt-1 text-2xl font-semibold">{value}</p>
            {hint && <p className="text-[11px] text-muted">{hint}</p>}
          </div>
        ))}
        <p className="col-span-2 text-xs text-muted">
          Release found by{" "}
          {d.release_source === "user"
            ? "you"
            : d.release_source === "model"
              ? "the learned model"
              : "rules"}
          . Wrong frame? Step to it in the player and press{" "}
          <span className="text-release">Set release here</span>.
        </p>
      </div>
    </div>
  );
}

function metric(m: BiomechMetric | undefined): string {
  return m && m.value !== null ? `${m.value.toFixed(0)}%` : "—";
}

/** Skeleton silhouettes at each phase, with the key angle; click to seek. */
export function PhaseFilmstrip({
  d,
  frames,
  arm,
  edges,
  onSeek,
}: {
  d: DeliveryMetrics;
  frames: Map<number, FramePose> | null;
  arm: Arm;
  edges: string[][];
  onSeek: (clipFrame: number) => void;
}) {
  const events = PHASE_EVENTS.filter(([k]) => k !== "run_up_start");
  const poses = events.map(([k]) => frames?.get(d.clip_start + d.phases[k]));
  const pts = poses.flatMap((p) => Object.values(p?.keypoints ?? {}).filter(Boolean)) as {
    x: number;
    y: number;
  }[];
  if (pts.length === 0) return <p className="text-sm text-muted">Loading skeletons…</p>;
  const ys = pts.map((p) => p.y);
  const top = Math.min(...ys);
  const span = Math.max(1, Math.max(...ys) - top);
  return (
    <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
      {events.map(([key, label], i) => {
        const p = poses[i];
        const a = frameAngles(p, arm);
        const xs = Object.values(p?.keypoints ?? {})
          .filter(Boolean)
          .map((k) => k!.x);
        const cx = xs.length ? (Math.min(...xs) + Math.max(...xs)) / 2 : 0;
        const view = `${cx - span * 0.55} ${top - span * 0.05} ${span * 1.1} ${span * 1.1}`;
        const isRel = key === "release";
        return (
          <button
            key={key}
            type="button"
            onClick={() => onSeek(d.phases[key])}
            className={cn(
              "rounded-sm border bg-surface-2 p-2 text-left hover:border-accent",
              isRel ? "border-release" : "border-border",
            )}
          >
            <svg viewBox={view} className="h-28 w-full" role="img" aria-label={`${label} skeleton`}>
              {edges.map(([a1, b1]) => {
                const q1 = p?.keypoints[a1];
                const q2 = p?.keypoints[b1];
                if (!q1 || !q2) return null;
                const bowl = [a1, b1].every(
                  (n) =>
                    n.startsWith(jointName(arm, "bowling", "")) && /shoulder|elbow|wrist/.test(n),
                );
                return (
                  <line
                    key={a1 + b1}
                    x1={q1.x}
                    y1={q1.y}
                    x2={q2.x}
                    y2={q2.y}
                    stroke={bowl ? "#c6f432" : isRel ? "#ff7a2f" : "#e9efe7"}
                    strokeWidth={span / 70}
                    strokeLinecap="round"
                  />
                );
              })}
            </svg>
            <p
              className={cn(
                "font-display text-[11px] uppercase tracking-widest",
                isRel ? "text-release" : "text-muted",
              )}
            >
              {label}
            </p>
            <p className="tabular text-xs">
              E {fmtDeg(a.elbow)} · K {fmtDeg(a.frontKnee)}
            </p>
          </button>
        );
      })}
    </div>
  );
}

/** All biomechanical measurements, grouped by moment of the delivery. */
export function MechanicsPanel({ metrics }: { metrics: BiomechMetric[] }) {
  const groups = [...new Set(metrics.map((m) => m.group))];
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {groups.map((g) => (
        <div key={g} className="rounded-sm border border-border bg-surface-2 p-3">
          <p className="mb-2 font-display text-xs uppercase tracking-widest text-accent">{g}</p>
          <table className="w-full text-sm">
            <tbody>
              {metrics
                .filter((m) => m.group === g)
                .map((m) => (
                  <tr
                    key={m.key}
                    className="border-t border-border/60 first:border-0"
                    title={m.description}
                  >
                    <td className="py-1.5 pr-2">{m.name}</td>
                    <td className="tabular py-1.5 text-right font-semibold">
                      {m.value === null ? (
                        <span
                          className="text-xs font-normal text-muted"
                          title={m.not_measured ?? ""}
                        >
                          n/a
                        </span>
                      ) : (
                        <>
                          {Math.abs(m.value) >= 100
                            ? m.value.toFixed(0)
                            : m.value.toFixed(m.unit === "s" ? 3 : 1)}
                          <span className="ml-1 text-xs font-normal text-muted">{m.unit}</span>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

/** Pelvis -> trunk -> arm peak timings before release (ideal: largest to smallest). */
export function KineticChain({ metrics }: { metrics: BiomechMetric[] }) {
  const parts = ["chain_pelvis", "chain_trunk", "chain_arm"].map((k) =>
    metrics.find((m) => m.key === k),
  );
  if (parts.some((p) => !p || p.value === null))
    return <p className="text-sm text-muted">Kinetic-chain timing needs a side-on clip.</p>;
  const max = Math.max(...parts.map((p) => p!.value!), 0.01);
  const ok = parts[0]!.value! >= parts[1]!.value! && parts[1]!.value! >= parts[2]!.value!;
  return (
    <div className="space-y-2">
      {parts.map((p, i) => (
        <div key={p!.key} className="grid grid-cols-[5rem_1fr_4.5rem] items-center gap-2 text-sm">
          <span className="font-display uppercase tracking-wider text-muted">
            {["Pelvis", "Trunk", "Arm"][i]}
          </span>
          <div className="relative h-3 rounded-sm bg-surface-2">
            <div
              className="absolute right-0 h-3 rounded-sm bg-accent"
              style={{ width: `${(p!.value! / max) * 100}%` }}
            />
          </div>
          <span className="tabular text-right">{(p!.value! * 1000).toFixed(0)} ms</span>
        </div>
      ))}
      <p className={cn("text-xs", ok ? "text-good" : "text-warn")}>
        {ok
          ? "Sequence OK: pelvis, then trunk, then arm peak in order before release."
          : "Out of order: the energy chain from hips to arm is broken."}
      </p>
    </div>
  );
}
