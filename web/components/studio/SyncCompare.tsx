"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { type Arm, frameAngles, jointName } from "@/lib/pose";
import type { DeliveryOut, FramePose } from "@/lib/types";
import { cn } from "@/lib/utils";

export interface Side {
  out: DeliveryOut;
  frames: Map<number, FramePose>;
  label: string;
}

const SPEEDS = [0.05, 0.1, 0.25, 1] as const;
const WINDOW: [number, number] = [-700, 450]; // ms around release

/** Video frame of ``side`` at ``ms`` from its release (in real time). */
function frameAt(side: Side, ms: number): number {
  const d = side.out.delivery;
  return Math.round(d.clip_start + d.phases.release + (ms / 1000) * side.out.analysis_fps);
}

function bodyScale(f: FramePose | undefined): number | null {
  const k = f?.keypoints;
  if (!k?.nose || !(k.l_ankle || k.r_ankle)) return null;
  const a = k.l_ankle ?? k.r_ankle!;
  return Math.hypot(k.nose.x - a.x, k.nose.y - a.y) || null;
}

function hipMid(f: FramePose | undefined): { x: number; y: number } | null {
  const k = f?.keypoints;
  return k?.l_hip && k?.r_hip
    ? { x: (k.l_hip.x + k.r_hip.x) / 2, y: (k.l_hip.y + k.r_hip.y) / 2 }
    : null;
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  f: FramePose | undefined,
  edges: string[][],
  arm: Arm,
  map: (x: number, y: number) => [number, number],
  colour: string,
  armColour: string,
  width: number,
) {
  if (!f) return;
  const k = f.keypoints;
  const bowl = new Set([
    jointName(arm, "bowling", "shoulder"),
    jointName(arm, "bowling", "elbow"),
    jointName(arm, "bowling", "wrist"),
  ]);
  ctx.lineCap = "round";
  for (const [a, b] of edges) {
    const p = k[a];
    const q = k[b];
    if (!p || !q) continue;
    const [x1, y1] = map(p.x, p.y);
    const [x2, y2] = map(q.x, q.y);
    ctx.strokeStyle = bowl.has(a) && bowl.has(b) ? armColour : colour;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
}

/** Two deliveries side by side, locked together at the release point, with the other
 *  bowler's skeleton ghosted over each video (scaled to the same body height). */
export function SyncCompare({ a, b, edges }: { a: Side; b: Side; edges: string[][] }) {
  const [ms, setMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(0.1);
  const [ghost, setGhost] = useState(true);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = now - last;
      last = now;
      setMs((m) => {
        const next = m + dt * speed;
        return next > WINDOW[1] ? WINDOW[0] : next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === "INPUT") return;
      if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        setPlaying(false);
        const step = 1000 / Math.max(a.out.analysis_fps, b.out.analysis_fps);
        setMs((m) => m + (e.key === "ArrowRight" ? step : -step));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [a.out.analysis_fps, b.out.analysis_fps]);

  const fa = a.frames.get(frameAt(a, ms));
  const fb = b.frames.get(frameAt(b, ms));
  const rows: [string, (f: FramePose | undefined, arm: Arm) => number | null][] = [
    ["Bowling elbow", (f, arm) => frameAngles(f, arm).elbow],
    ["Front knee", (f, arm) => frameAngles(f, arm).frontKnee],
    ["Trunk lean", (f, arm) => frameAngles(f, arm).trunkLean],
    ["Upper arm", (f, arm) => frameAngles(f, arm).upperArm],
  ];

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <Pane side={a} other={b} ms={ms} edges={edges} ghost={ghost} tag="A" />
        <Pane side={b} other={a} ms={ms} edges={edges} ghost={ghost} tag="B" />
      </div>

      <div className="flex flex-wrap items-center gap-4 border-y border-foreground py-3 text-sm">
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="min-w-24 bg-foreground px-4 py-2 text-background hover:bg-accent"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          min={WINDOW[0]}
          max={WINDOW[1]}
          step={1}
          value={Math.round(ms)}
          onChange={(e) => {
            setPlaying(false);
            setMs(Number(e.target.value));
          }}
          aria-label="Time from release"
          className="min-w-48 flex-1"
        />
        <span className="tabular w-24 text-right">
          {ms >= 0 ? "+" : ""}
          {Math.round(ms)} ms
        </span>
        <button
          type="button"
          onClick={() => {
            setPlaying(false);
            setMs(0);
          }}
          className="text-release hover:underline"
        >
          Release
        </button>
        <span className="flex items-center gap-1">
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSpeed(s)}
              className={cn(
                "px-2 py-1",
                speed === s ? "bg-foreground text-background" : "text-muted hover:text-foreground",
              )}
            >
              {s}×
            </button>
          ))}
        </span>
        <label className="flex items-center gap-2 text-muted">
          <input type="checkbox" checked={ghost} onChange={(e) => setGhost(e.target.checked)} />
          Ghost skeleton
        </label>
      </div>

      <table className="w-full text-sm">
        <thead>
          <tr className="kicker text-left">
            <th className="py-2 font-semibold">At {Math.round(ms)} ms</th>
            <th className="py-2 text-right font-semibold">A · {a.label}</th>
            <th className="py-2 text-right font-semibold">B · {b.label}</th>
            <th className="py-2 text-right font-semibold">Difference</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border border-y border-border">
          {rows.map(([name, fn]) => {
            const va = fn(fa, a.out.bowling_arm as Arm);
            const vb = fn(fb, b.out.bowling_arm as Arm);
            const diff = va !== null && vb !== null ? vb - va : null;
            return (
              <tr key={name}>
                <td className="py-2.5">{name}</td>
                <td className="tabular py-2.5 text-right">
                  {va === null ? "—" : `${va.toFixed(0)}°`}
                </td>
                <td className="tabular py-2.5 text-right">
                  {vb === null ? "—" : `${vb.toFixed(0)}°`}
                </td>
                <td
                  className={cn(
                    "tabular py-2.5 text-right",
                    diff !== null && Math.abs(diff) > 10 ? "text-accent" : "text-muted",
                  )}
                >
                  {diff === null ? "—" : `${diff > 0 ? "+" : ""}${diff.toFixed(0)}°`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Pane({
  side,
  other,
  ms,
  edges,
  ghost,
  tag,
}: {
  side: Side;
  other: Side;
  ms: number;
  edges: string[][];
  ghost: boolean;
  tag: string;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const { out } = side;
  const frame = frameAt(side, ms);

  const paint = useCallback(() => {
    const c = canvas.current;
    if (!c) return;
    c.width = out.width;
    c.height = out.height;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
    const u = Math.max(out.width, out.height) / 500;
    const mine = side.frames.get(frame);
    if (ghost) {
      const theirs = other.frames.get(frameAt(other, ms));
      const h1 = hipMid(mine);
      const h2 = hipMid(theirs);
      const s1 = bodyScale(side.frames.get(frameAt(side, 0)));
      const s2 = bodyScale(other.frames.get(frameAt(other, 0)));
      if (h1 && h2 && s1 && s2) {
        const k = s1 / s2;
        const flip = side.out.bowling_arm !== other.out.bowling_arm ? -1 : 1;
        drawSkeleton(ctx, theirs, edges, other.out.bowling_arm as Arm,
          (x, y) => [h1.x + flip * (x - h2.x) * k, h1.y + (y - h2.y) * k],
          "rgba(120,200,255,0.55)", "rgba(120,200,255,0.9)", 3 * u); // prettier-ignore
      }
    }
    drawSkeleton(
      ctx,
      mine,
      edges,
      out.bowling_arm as Arm,
      (x, y) => [x, y],
      "rgba(242,238,229,0.9)",
      "#ff5b3a",
      3 * u,
    );
  }, [out, side, other, frame, ms, edges, ghost]);

  useEffect(() => {
    const v = video.current;
    if (v && v.readyState >= 1 && !v.seeking) {
      const t = (frame + 0.5) / out.fps;
      if (Math.abs(v.currentTime - t) > 0.5 / out.fps) v.currentTime = Math.max(0, t);
    }
    paint();
  }, [frame, out.fps, paint]);

  return (
    <figure>
      <div className="relative overflow-hidden bg-plate">
        <video
          ref={video}
          src={api.videoUrl(out.session_id)}
          muted
          playsInline
          preload="auto"
          className="block max-h-[56vh] w-full object-contain"
          onLoadedMetadata={paint}
        />
        <canvas
          ref={canvas}
          className="pointer-events-none absolute inset-0 h-full w-full"
          style={{ objectFit: "contain" }}
        />
        <span className="absolute left-3 top-3 bg-black/70 px-2 py-1 font-display text-sm text-[#f2eee5]">
          {tag}
        </span>
        {Math.abs(ms) < 1000 / out.analysis_fps && (
          <span className="absolute right-3 top-3 bg-release px-2 py-1 text-xs font-semibold text-white">
            RELEASE
          </span>
        )}
      </div>
      <figcaption className="mt-2 flex justify-between text-sm">
        <span className="truncate">{side.label}</span>
        <span className="tabular text-muted">f {frame}</span>
      </figcaption>
    </figure>
  );
}
