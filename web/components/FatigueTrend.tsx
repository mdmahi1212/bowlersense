"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FatigueResult } from "@/lib/types";

const SERIES = [
  ["runup_speed_pct", "Run-up speed", "#2f5d8a"],
  ["release_height_pct", "Release height", "#a8670d"],
  ["wrist_speed_pct", "Wrist speed", "#b3261e"],
  ["technique_score_pct", "Technique score", "#16140f"],
] as const;

/** % change of each tracked metric from the baseline deliveries. */
export function FatigueTrend({ fatigue }: { fatigue: FatigueResult }) {
  if (fatigue.drifts.length === 0) {
    return (
      <p className="text-sm text-muted">
        {fatigue.notes.join(" ") || "Fatigue trends need more deliveries than the baseline."}
      </p>
    );
  }
  const data = fatigue.drifts.map((d) => ({ ...d, ball: d.delivery_idx + 1 }));
  return (
    <div
      className="h-56 w-full"
      role="img"
      aria-label="Fatigue trend: change from baseline per delivery"
    >
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -10 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis
            dataKey="ball"
            stroke="var(--muted)"
            fontSize={12}
            label={{
              value: "Ball",
              position: "insideBottomRight",
              offset: -2,
              fill: "var(--muted)",
              fontSize: 11,
            }}
          />
          <YAxis stroke="var(--muted)" fontSize={12} unit="%" />
          <Tooltip
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine y={0} stroke="var(--muted)" />
          {SERIES.map(([key, name, color]) => (
            <Line
              key={key}
              dataKey={key}
              name={name}
              stroke={color}
              dot={false}
              connectNulls
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
