"use client";

import { useMemo } from "react";
import { fileUrl } from "@/lib/api";
import { type XY, bodyHeightPx, releasePoint, spread } from "@/lib/biomech";
import { type Arm, frameIndex } from "@/lib/pose";
import type { DeliveryMetrics, PoseSequence } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";

/** Every release point of a session on one plot (bowling wrist relative to the hips, in % of
 *  height), with the spread converted to centimetres using the bowler's height. */
export function ReleaseConsistency({
  posePath,
  deliveries,
  arm,
  heightCm,
}: {
  posePath: string;
  deliveries: DeliveryMetrics[];
  arm: Arm;
  heightCm: number | null;
}) {
  const pose = useAsync(
    () =>
      fetch(fileUrl(posePath)).then((r) =>
        r.ok
          ? (r.json() as Promise<PoseSequence>)
          : Promise.reject(new Error(`skeleton ${r.status}`)),
      ),
    [posePath],
  );
  const pts = useMemo(() => {
    if (!pose.data) return [];
    const frames = frameIndex(pose.data);
    const h = bodyHeightPx(frames);
    if (!h) return [];
    return deliveries
      .map((d) => ({
        i: d.delivery_idx,
        p: releasePoint(frames.get(d.clip_start + d.phases.release), arm, h),
      }))
      .filter((x): x is { i: number; p: XY } => x.p !== null);
  }, [pose.data, deliveries, arm]);
  if (pose.error) return <p className="text-sm text-muted">Skeleton unavailable: {pose.error}</p>;
  if (!pose.data) return <p className="text-sm text-muted">Loading release points…</p>;
  if (pts.length === 0)
    return <p className="text-sm text-muted">No release points could be measured.</p>;

  const s = spread(pts.map((x) => x.p))!;
  const cm = (heightCm ?? 175) / 100;
  const xs = pts.map((x) => x.p.x);
  const ys = pts.map((x) => x.p.y);
  const pad = 6;
  const x0 = Math.min(...xs, s.mean.x - 8) - pad;
  const x1 = Math.max(...xs, s.mean.x + 8) + pad;
  const y0 = Math.min(...ys, s.mean.y - 8) - pad;
  const y1 = Math.max(...ys, s.mean.y + 8) + pad;
  const W = 320;
  const H = Math.round((W * (y1 - y0)) / (x1 - x0));
  const px = (x: number) => ((x - x0) / (x1 - x0)) * W;
  const py = (y: number) => H - ((y - y0) / (y1 - y0)) * H;

  return (
    <div className="grid items-center gap-8 sm:grid-cols-[minmax(0,320px)_1fr]">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full border border-border bg-surface"
        role="img"
        aria-label="Release points of every delivery"
      >
        {[...Array(9)].map((_, i) => (
          <line
            key={i}
            x1={(i * W) / 8}
            y1={0}
            x2={(i * W) / 8}
            y2={H}
            stroke="var(--border)"
            strokeWidth={0.5}
          />
        ))}
        <ellipse
          cx={px(s.mean.x)}
          cy={py(s.mean.y)}
          rx={Math.max(3, (s.sd.x / (x1 - x0)) * W * 2)}
          ry={Math.max(3, (s.sd.y / (y1 - y0)) * H * 2)}
          fill="var(--accent)"
          fillOpacity={0.08}
          stroke="var(--accent)"
          strokeDasharray="3 3"
        />
        {pts.map(({ i, p }) => (
          <g key={i} data-reveal style={{ "--d": `${i * 80}ms` } as React.CSSProperties}>
            <circle cx={px(p.x)} cy={py(p.y)} r={5} fill="var(--foreground)" />
            <text
              x={px(p.x) + 8}
              y={py(p.y) + 4}
              fontSize={10}
              fill="var(--muted)"
              fontFamily="var(--font-mono)"
            >
              {i + 1}
            </text>
          </g>
        ))}
        <circle cx={px(s.mean.x)} cy={py(s.mean.y)} r={3} fill="var(--accent)" />
      </svg>
      <dl className="grid grid-cols-2 gap-6">
        <div>
          <dt className="kicker">Spread (2 SD)</dt>
          <dd className="mt-1 font-display text-5xl leading-none">
            {(s.radius * 2 * cm).toFixed(0)}
            <span className="ml-1 text-base text-muted">cm</span>
          </dd>
        </div>
        <div>
          <dt className="kicker">Average height above hips</dt>
          <dd className="mt-1 font-display text-5xl leading-none">
            {(s.mean.y * cm).toFixed(0)}
            <span className="ml-1 text-base text-muted">cm</span>
          </dd>
        </div>
        <p className="col-span-2 text-sm text-muted">
          Each dot is where the bowling hand let go of the ball, measured from the hips.{" "}
          {pts.length < 2
            ? "Bowl more deliveries in one session to see the spread."
            : "A tight cluster means a repeatable release, which helps accuracy."}
          {heightCm === null &&
            " Centimetres assume a 175 cm bowler: add a height to the profile for exact values."}
        </p>
      </dl>
    </div>
  );
}
