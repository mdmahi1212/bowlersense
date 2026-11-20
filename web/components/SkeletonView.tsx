import type { FramePose } from "@/lib/types";

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** Bounding box of every visible keypoint in a set of frames. */
export function boundsOf(frames: FramePose[]): Bounds {
  const b = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (const f of frames) {
    for (const k of Object.values(f.keypoints)) {
      if (!k) continue;
      b.minX = Math.min(b.minX, k.x);
      b.maxX = Math.max(b.maxX, k.x);
      b.minY = Math.min(b.minY, k.y);
      b.maxY = Math.max(b.maxY, k.y);
    }
  }
  return Number.isFinite(b.minX) ? b : { minX: 0, minY: 0, maxX: 1, maxY: 1 };
}

/** One pose frame drawn as a stick figure inside fixed bounds (so motion stays visible). */
export function SkeletonView({
  frame,
  bounds,
  edges,
  colour,
  mirror = false,
  label,
}: {
  frame: FramePose | null;
  bounds: Bounds;
  edges: string[][];
  colour: string;
  mirror?: boolean;
  label: string;
}) {
  const pad = 0.06 * Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY);
  const w = bounds.maxX - bounds.minX + 2 * pad;
  const h = bounds.maxY - bounds.minY + 2 * pad;
  const px = (x: number) => (mirror ? bounds.maxX + pad - x : x - bounds.minX + pad);
  const py = (y: number) => y - bounds.minY + pad;
  const kp = frame?.keypoints ?? {};
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="h-72 w-full rounded-md bg-surface-2"
      role="img"
      aria-label={label}
    >
      {edges.map(([a, b]) => {
        const p = kp[a];
        const q = kp[b];
        if (!p || !q) return null;
        return (
          <line
            key={`${a}-${b}`}
            x1={px(p.x)}
            y1={py(p.y)}
            x2={px(q.x)}
            y2={py(q.y)}
            stroke={colour}
            strokeWidth={w / 120}
            strokeLinecap="round"
          />
        );
      })}
      {Object.entries(kp).map(([name, k]) =>
        k ? (
          <circle key={name} cx={px(k.x)} cy={py(k.y)} r={w / 140} fill="var(--foreground)" />
        ) : null,
      )}
    </svg>
  );
}
