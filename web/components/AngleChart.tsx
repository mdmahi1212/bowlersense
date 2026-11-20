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
import type { AngleSeries } from "@/lib/types";

/** One angle over the delivery, synced to the player (vertical line) and click-to-seek. */
export function AngleChart({
  series,
  title,
  current,
  markers,
  band,
  onSeek,
}: {
  series: AngleSeries;
  title: string;
  current: number;
  markers: { frame: number; label: string }[];
  band?: { from: number; to: number; label: string };
  onSeek: (frame: number) => void;
}) {
  const data = series.frames.map((f, i) => ({ frame: f, value: series.values[i] }));
  const has = series.values.some((v) => v !== null);
  return (
    <div>
      <p className="mb-1 text-sm font-medium">{title}</p>
      {!has ? (
        <p className="text-sm text-muted">Not measurable from this camera view.</p>
      ) : (
        <div className="h-44 w-full" role="img" aria-label={`${title} over the delivery`}>
          <ResponsiveContainer>
            <LineChart
              data={data}
              margin={{ top: 6, right: 8, bottom: 0, left: -18 }}
              onClick={(e) => {
                const label = (e as { activeLabel?: number | string } | null)?.activeLabel;
                if (label !== undefined) onSeek(Number(label));
              }}
            >
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis
                dataKey="frame"
                type="number"
                domain={["dataMin", "dataMax"]}
                stroke="var(--muted)"
                fontSize={11}
              />
              <YAxis stroke="var(--muted)" fontSize={11} unit="°" domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }}
                formatter={(v) => [`${Number(v).toFixed(1)}°`, title]}
                labelFormatter={(l) => `Frame ${l}`}
              />
              {band && (
                <ReferenceArea
                  y1={band.from}
                  y2={band.to}
                  fill="var(--warn)"
                  fillOpacity={0.15}
                  label={{
                    value: band.label,
                    fontSize: 10,
                    fill: "var(--warn)",
                    position: "insideTopLeft",
                  }}
                />
              )}
              {markers.map((m) => (
                <ReferenceLine
                  key={m.label}
                  x={m.frame}
                  stroke="var(--muted)"
                  strokeDasharray="4 3"
                  label={{ value: m.label, fontSize: 9, fill: "var(--muted)", position: "top" }}
                />
              ))}
              <ReferenceLine x={current} stroke="var(--warn)" strokeWidth={2} />
              <Line
                dataKey="value"
                stroke="var(--accent)"
                dot={false}
                strokeWidth={2}
                connectNulls={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
