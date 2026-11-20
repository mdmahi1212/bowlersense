"use client";

import { useEffect, useRef, useState } from "react";
import { type Arm, angleAt, fmtDeg, frameAngles, jointName } from "@/lib/pose";
import type { FramePose } from "@/lib/types";
import { cn } from "@/lib/utils";

const RATES = [0.0625, 0.125, 0.25] as const;

/** Auto-repeating slow-motion loop around the release with onion-skin ghosts of the bowling
 *  arm, the wrist path, and a frame-by-frame release sequence. */
export function ReleaseLoop({
  videoSrc,
  fps,
  realFps,
  releaseFrame,
  frames,
  arm,
  edges,
  width,
  height,
}: {
  videoSrc: string;
  fps: number;
  realFps: number;
  releaseFrame: number; // video frame
  frames: Map<number, FramePose> | null;
  arm: Arm;
  edges: string[][];
  width: number;
  height: number;
}) {
  const half = Math.max(3, Math.round(0.12 * realFps)); // +-120 ms real time
  const lo = Math.max(0, releaseFrame - half);
  const hi = releaseFrame + half;
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [rate, setRate] = useState<number>(0.125);
  const [ghosts, setGhosts] = useState(true);
  const [current, setCurrent] = useState(releaseFrame);
  const [paused, setPaused] = useState(false);

  function paint(f: number) {
    const c = canvas.current;
    if (!c || !frames) return;
    if (c.width !== width) {
      c.width = width;
      c.height = height;
    }
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    const s = Math.max(1.5, width / 500);
    const wristName = jointName(arm, "bowling", "wrist");
    const chain = ["shoulder", "elbow", "wrist"].map((p) => jointName(arm, "bowling", p));
    // wrist path over the whole loop window
    ctx.setLineDash([6 * s, 5 * s]);
    ctx.strokeStyle = "rgba(198,244,50,0.55)";
    ctx.lineWidth = 1.8 * s;
    ctx.beginPath();
    let pen = false;
    for (let i = lo; i < hi; i++) {
      const w = frames.get(i)?.keypoints[wristName];
      if (!w) {
        pen = false;
        continue;
      }
      if (pen) ctx.lineTo(w.x, w.y);
      else ctx.moveTo(w.x, w.y);
      pen = true;
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // onion skin: the bowling arm in earlier frames, fading
    if (ghosts) {
      const step = Math.max(1, Math.round(half / 6));
      for (let i = Math.max(lo, f - step * 6); i < f; i += step) {
        const k = frames.get(i)?.keypoints;
        if (!k) continue;
        const alpha = 0.12 + 0.5 * (1 - (f - i) / (step * 6));
        ctx.strokeStyle = `rgba(198,244,50,${alpha.toFixed(2)})`;
        ctx.lineWidth = 2.4 * s;
        ctx.beginPath();
        chain.forEach((n, j) => {
          const p = k[n];
          if (p && j) ctx.lineTo(p.x, p.y);
          else if (p) ctx.moveTo(p.x, p.y);
        });
        ctx.stroke();
      }
    }
    // current skeleton
    const k = frames.get(f)?.keypoints;
    if (k) {
      for (const [a, b] of edges) {
        const p = k[a];
        const q = k[b];
        if (!p || !q) continue;
        const arm3 = chain.includes(a) && chain.includes(b);
        ctx.strokeStyle = arm3 ? "#c6f432" : "rgba(255,255,255,0.75)";
        ctx.lineWidth = (arm3 ? 3.4 : 2) * s;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(q.x, q.y);
        ctx.stroke();
      }
    }
    // release position of the wrist
    const r = frames.get(releaseFrame)?.keypoints[wristName];
    if (r) {
      ctx.strokeStyle = "#ff7a2f";
      ctx.lineWidth = 2 * s;
      ctx.beginPath();
      ctx.arc(r.x, r.y, 9 * s, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  useEffect(() => {
    const v = video.current as
      | (HTMLVideoElement & {
          requestVideoFrameCallback?: (cb: (n: number, m: { mediaTime: number }) => void) => number;
          cancelVideoFrameCallback?: (h: number) => void;
        })
      | null;
    if (!v) return;
    let handle = 0;
    let raf = 0;
    const start = () => {
      v.currentTime = (lo + 0.5) / fps;
      v.playbackRate = rate;
      if (!paused) void v.play().catch(() => undefined);
    };
    const onTime = (t: number) => {
      const f = Math.floor(t * fps + 1e-3);
      if (f >= hi) {
        v.currentTime = (lo + 0.5) / fps;
        return;
      }
      setCurrent(f);
      paint(f);
    };
    if (v.readyState >= 1) start();
    else v.addEventListener("loadedmetadata", start, { once: true });
    if (v.requestVideoFrameCallback) {
      const tick = (_: number, m: { mediaTime: number }) => {
        onTime(m.mediaTime);
        handle = v.requestVideoFrameCallback!(tick);
      };
      handle = v.requestVideoFrameCallback(tick);
    } else {
      const loop = () => {
        onTime(v.currentTime);
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    }
    return () => {
      v.cancelVideoFrameCallback?.(handle);
      cancelAnimationFrame(raf);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lo, hi, fps, rate, paused, frames, ghosts]);

  const ms = ((current - releaseFrame) / realFps) * 1000;
  const a = frameAngles(frames?.get(current), arm);
  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-md border border-release/60 bg-black">
        <video
          ref={video}
          src={videoSrc}
          muted
          playsInline
          preload="auto"
          className="block max-h-[38vh] w-full object-contain"
          onClick={() => setPaused((p) => !p)}
        />
        <canvas
          ref={canvas}
          className="pointer-events-none absolute inset-0 h-full w-full object-contain"
        />
        <div className="pointer-events-none absolute left-2 top-2 flex gap-1.5">
          <span className="rounded-sm bg-release px-1.5 py-0.5 font-display text-[10px] font-bold uppercase tracking-widest text-black">
            Release loop
          </span>
          <span className="rounded-sm bg-black/70 px-1.5 py-0.5 text-[10px] text-foreground">
            ±{Math.round((half / realFps) * 1000)} ms · {Math.round(rate * 100)}% speed
          </span>
        </div>
        <div className="tabular pointer-events-none absolute bottom-2 left-2 rounded-sm bg-black/75 px-2 py-1 text-xs">
          <span className={cn(Math.abs(ms) < 1000 / realFps ? "text-release" : "text-foreground")}>
            {ms >= 0 ? "+" : ""}
            {ms.toFixed(0)} ms
          </span>
          <span className="ml-3 text-accent">elbow {fmtDeg(a.elbow)}</span>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <button
          type="button"
          onClick={() => setPaused((p) => !p)}
          className="rounded-sm border border-border px-2 py-1 hover:border-accent"
        >
          {paused ? "▶ Loop" : "❚❚ Hold"}
        </button>
        {RATES.map((r) => (
          <button
            key={r}
            type="button"
            onClick={() => setRate(r)}
            className={cn(
              "tabular rounded-sm border px-2 py-1",
              rate === r ? "border-accent text-accent" : "border-border hover:border-accent",
            )}
          >
            {r < 0.1 ? "1/16" : r < 0.2 ? "1/8" : "1/4"}×
          </button>
        ))}
        <label className="ml-auto flex items-center gap-1.5 text-muted">
          <input type="checkbox" checked={ghosts} onChange={(e) => setGhosts(e.target.checked)} />{" "}
          Onion skin
        </label>
      </div>
      <ReleaseSequence
        frames={frames}
        arm={arm}
        edges={edges}
        releaseFrame={releaseFrame}
        half={half}
        realFps={realFps}
        current={current}
      />
    </div>
  );
}

/** Nine skeleton snapshots across the release window with the elbow angle under each. */
function ReleaseSequence({
  frames,
  arm,
  edges,
  releaseFrame,
  half,
  realFps,
  current,
}: {
  frames: Map<number, FramePose> | null;
  arm: Arm;
  edges: string[][];
  releaseFrame: number;
  half: number;
  realFps: number;
  current: number;
}) {
  const step = Math.max(1, Math.round(half / 4));
  const list = Array.from({ length: 9 }, (_, i) => releaseFrame + (i - 4) * step);
  const all = list.flatMap((f) =>
    Object.values(frames?.get(f)?.keypoints ?? {}).filter(Boolean),
  ) as { x: number; y: number }[];
  if (!all.length) return null;
  const minX = Math.min(...all.map((p) => p.x));
  const maxX = Math.max(...all.map((p) => p.x));
  const minY = Math.min(...all.map((p) => p.y));
  const maxY = Math.max(...all.map((p) => p.y));
  const size = Math.max(maxX - minX, maxY - minY) * 1.08;
  const chain = ["shoulder", "elbow", "wrist"].map((p) => jointName(arm, "bowling", p));
  return (
    <div className="grid grid-cols-9 gap-1">
      {list.map((f) => {
        const k = frames?.get(f)?.keypoints ?? {};
        const g = (p: string) => k[jointName(arm, "bowling", p)];
        const elbow = angleAt(g("shoulder"), g("elbow"), g("wrist"));
        const isRel = f === releaseFrame;
        const near = Math.abs(current - f) < step / 2;
        return (
          <div
            key={f}
            className={cn(
              "rounded-sm border bg-surface-2 p-1 text-center",
              isRel ? "border-release" : near ? "border-accent" : "border-border",
            )}
          >
            <svg
              viewBox={`${(minX + maxX) / 2 - size / 2} ${(minY + maxY) / 2 - size / 2} ${size} ${size}`}
              className="aspect-square w-full"
              aria-hidden="true"
            >
              {edges.map(([a, b]) => {
                const p = k[a];
                const q = k[b];
                if (!p || !q) return null;
                const arm3 = chain.includes(a) && chain.includes(b);
                return (
                  <line
                    key={a + b}
                    x1={p.x}
                    y1={p.y}
                    x2={q.x}
                    y2={q.y}
                    stroke={arm3 ? "#c6f432" : "#9aa8a2"}
                    strokeWidth={size / (arm3 ? 45 : 70)}
                    strokeLinecap="round"
                  />
                );
              })}
            </svg>
            <p className={cn("tabular text-[10px]", isRel ? "text-release" : "text-muted")}>
              {(((f - releaseFrame) / realFps) * 1000).toFixed(0)}ms
            </p>
            <p className="tabular text-[11px] font-semibold">{fmtDeg(elbow)}</p>
          </div>
        );
      })}
    </div>
  );
}
