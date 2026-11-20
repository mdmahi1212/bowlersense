"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { WorkloadOut } from "@/lib/types";

/** Acute:chronic workload ratio over time with the 0.8-1.3 "safe" band. */
export function WorkloadChart({ workload }: { workload: WorkloadOut }) {
  const data = workload.points.map((p) => ({ day: p.day, acwr: p.acwr, balls: p.balls }));
  if (!data.some((p) => p.acwr !== null)) {
    return (
      <p className="text-sm text-muted">
        {workload.latest?.note || "The workload ratio needs about four weeks of sessions."}
      </p>
    );
  }
  const [lo, hi] = workload.safe_band;
  return (
    <div className="h-56 w-full" role="img" aria-label="Acute to chronic workload ratio over time">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="day" stroke="var(--muted)" fontSize={11} />
          <YAxis stroke="var(--muted)" fontSize={11} domain={[0, "auto"]} />
          <Tooltip
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }}
          />
          <ReferenceArea
            y1={lo}
            y2={hi}
            fill="var(--good)"
            fillOpacity={0.12}
            label={{
              value: "safe band",
              fontSize: 10,
              fill: "var(--good)",
              position: "insideTopLeft",
            }}
          />
          <ReferenceLine
            y={workload.flag_above}
            stroke="var(--bad)"
            strokeDasharray="4 3"
            label={{ value: "high", fontSize: 10, fill: "var(--bad)", position: "right" }}
          />
          <Line dataKey="acwr" name="ACWR" stroke="var(--accent)" strokeWidth={2} connectNulls />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
