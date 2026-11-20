import type { DeliveryMetrics } from "@/lib/types";

const STUMPS_FROM_POPPING_M = 17.68 + 1.22; // bowler's popping crease -> batter's stumps
const ZONES = [
  ["Yorker", 0, 2, "#5f584b"],
  ["Full", 2, 6, "#2f5d8a"],
  ["Good", 6, 8, "#2e7048"],
  ["Short", 8, 10, "#a8670d"],
  ["Bouncer", 10, 14, "#b3261e"],
] as const;
const W = 160;
const H = 300;
const M_PER_PX = 14 / (H - 20);
const toY = (m: number) => 10 + m / M_PER_PX;
const toX = (lateral: number) => W / 2 - lateral / M_PER_PX; // + = bowler's left (drawn left)

/** Top-down pitch (batter's stumps at the top) with length zones and each ball's bounce. */
export function PitchMap({ deliveries }: { deliveries: DeliveryMetrics[] }) {
  const points = deliveries
    .filter((d) => d.pitch_xy)
    .map((d) => ({
      idx: d.delivery_idx,
      along: STUMPS_FROM_POPPING_M - d.pitch_xy![0],
      lat: d.pitch_xy![1],
      length: d.length,
    }));
  if (points.length === 0) {
    return (
      <p className="text-sm text-muted">
        No bounce positions yet. Calibrate the pitch (click the four crease corners) to measure
        speed, length and line.
      </p>
    );
  }
  return (
    <figure className="flex flex-col items-center">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-72"
        role="img"
        aria-label={`Pitch map with ${points.length} deliveries`}
      >
        {ZONES.map(([name, a, b, color]) => (
          <g key={name}>
            <rect
              x={W / 2 - 1.5 / M_PER_PX}
              y={toY(a)}
              width={3 / M_PER_PX}
              height={(b - a) / M_PER_PX}
              fill={color}
              opacity={0.18}
            />
            <text x={4} y={toY((a + b) / 2) + 3} fontSize={9} fill="var(--muted)">
              {name}
            </text>
          </g>
        ))}
        <rect
          x={W / 2 - 0.115 / M_PER_PX}
          y={4}
          width={0.23 / M_PER_PX}
          height={6}
          fill="var(--foreground)"
        />
        {points.map((p) => (
          <g key={p.idx}>
            <circle
              cx={toX(p.lat)}
              cy={toY(Math.max(0, p.along))}
              r={5}
              fill="var(--accent)"
              stroke="var(--background)"
            />
            <text
              x={toX(p.lat) + 7}
              y={toY(Math.max(0, p.along)) + 3}
              fontSize={9}
              fill="var(--foreground)"
            >
              {p.idx + 1}
            </text>
          </g>
        ))}
      </svg>
      <figcaption className="text-xs text-muted">
        Batter&apos;s stumps at the top · numbers are ball order
      </figcaption>
    </figure>
  );
}
