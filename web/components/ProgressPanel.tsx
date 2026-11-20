"use client";

import Link from "next/link";
import { useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ProgressOut, ProgressPoint } from "@/lib/types";
import { cn, fmtDate } from "@/lib/utils";

type Key =
  | "technique"
  | "injury"
  | "elbow_extension_max"
  | "release_height_pct"
  | "stride_pct"
  | "arm_speed_deg_s";

const METRICS: {
  key: Key;
  label: string;
  unit: string;
  better: "up" | "down" | null;
  ref?: number;
}[] = [
  { key: "technique", label: "Technique", unit: "/100", better: "up" },
  { key: "injury", label: "Injury load", unit: "/100", better: "down" },
  { key: "elbow_extension_max", label: "Elbow straightening", unit: "°", better: "down", ref: 15 },
  { key: "release_height_pct", label: "Release height", unit: "% ht", better: "up" },
  { key: "stride_pct", label: "Stride", unit: "% ht", better: null },
  { key: "arm_speed_deg_s", label: "Arm speed", unit: "°/s", better: "up" },
];

const val = (p: ProgressPoint, k: Key) => p[k] as number | null;

function Spark({ values }: { values: (number | null)[] }) {
  const xs = values.filter((v): v is number => v !== null);
  if (xs.length < 2) return <svg className="h-6 w-full" />;
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  const pts = values
    .map((v, i) =>
      v === null
        ? null
        : `${(i / (values.length - 1)) * 100},${22 - ((v - lo) / (hi - lo || 1)) * 20}`,
    )
    .filter(Boolean)
    .join(" ");
  return (
    <svg viewBox="0 0 100 24" preserveAspectRatio="none" className="h-6 w-full">
      <polyline
        points={pts}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

/** A bowler's numbers across sessions: a strip of sparklines, a large chart for the chosen
 *  measure, the change since the first session and the best and worst deliveries. */
export function ProgressPanel({ progress }: { progress: ProgressOut }) {
  const [key, setKey] = useState<Key>("technique");
  const pts = progress.points;
  if (pts.length === 0) {
    return (
      <p className="text-muted">
        No finished sessions yet. Analyse a video for this bowler to start a history.
      </p>
    );
  }
  const m = METRICS.find((x) => x.key === key)!;
  const data = pts.map((p, i) => ({
    i: i + 1,
    day: fmtDate(p.day),
    v: val(p, key),
    id: p.session_id,
  }));
  const first = data.find((d) => d.v !== null)?.v ?? null;
  const last = [...data].reverse().find((d) => d.v !== null)?.v ?? null;
  const change = first !== null && last !== null ? last - first : null;
  const good =
    change === null || m.better === null ? null : m.better === "up" ? change > 0 : change < 0;

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 border-y border-border sm:grid-cols-3 xl:grid-cols-6">
        {METRICS.map((x, i) => {
          const series = pts.map((p) => val(p, x.key));
          const latest = [...series].reverse().find((v) => v !== null) ?? null;
          return (
            <button
              key={x.key}
              type="button"
              data-reveal
              style={{ "--d": `${i * 60}ms` } as React.CSSProperties}
              onClick={() => setKey(x.key)}
              aria-pressed={key === x.key}
              className={cn(
                "border-border px-4 py-3 text-left transition-colors [&:not(:last-child)]:border-r",
                key === x.key ? "bg-foreground text-background" : "hover:bg-surface-2",
              )}
            >
              <span className={cn("kicker", key === x.key && "!text-background/70")}>
                {x.label}
              </span>
              <span className="mt-1 block font-display text-3xl leading-none">
                {latest === null ? "—" : latest.toFixed(x.key === "arm_speed_deg_s" ? 0 : 1)}
                <span className="ml-1 text-xs opacity-60">{x.unit}</span>
              </span>
              <span className={cn("mt-2 block", key === x.key ? "text-accent" : "text-muted")}>
                <Spark values={series} />
              </span>
            </button>
          );
        })}
      </div>

      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_260px]">
        <div className="h-72">
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="var(--border)" vertical={false} />
              <XAxis dataKey="day" stroke="var(--muted)" fontSize={11} tickLine={false} />
              <YAxis stroke="var(--muted)" fontSize={11} tickLine={false} axisLine={false} />
              {m.ref !== undefined && (
                <ReferenceLine
                  y={m.ref}
                  stroke="var(--accent)"
                  strokeDasharray="4 4"
                  label={{
                    value: `${m.ref}° limit`,
                    fill: "var(--accent)",
                    fontSize: 11,
                    position: "insideTopRight",
                  }}
                />
              )}
              <Tooltip
                contentStyle={{
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: 0,
                }}
                formatter={(v) => [`${Number(v).toFixed(1)} ${m.unit}`, m.label]}
              />
              <Line
                type="monotone"
                dataKey="v"
                stroke="var(--foreground)"
                strokeWidth={2}
                dot={{ r: 4, fill: "var(--background)", strokeWidth: 2 }}
                activeDot={{ r: 6, fill: "var(--accent)" }}
                connectNulls
                animationDuration={900}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-6">
          <div>
            <p className="kicker">Since the first session</p>
            <p
              className={cn(
                "mt-2 font-display text-5xl leading-none",
                good === null ? "" : good ? "text-good" : "text-bad",
              )}
            >
              {change === null ? "—" : `${change > 0 ? "+" : ""}${change.toFixed(1)}`}
              <span className="ml-1 text-base text-muted">{m.unit}</span>
            </p>
            <p className="mt-2 text-sm text-muted">
              {pts.length} session{pts.length === 1 ? "" : "s"},{" "}
              {pts.reduce((a, p) => a + p.deliveries, 0)} deliveries
              {good !== null && (good ? " · improving" : " · getting worse")}
            </p>
          </div>
          {progress.best && (
            <Link
              href={`/deliveries/${progress.best.delivery_id}`}
              className="lift block border-t border-border pt-3"
            >
              <span className="kicker">Best delivery</span>
              <span className="mt-1 block font-display text-xl">
                Technique {progress.best.technique.toFixed(0)} →
              </span>
            </Link>
          )}
          {progress.worst && progress.worst.delivery_id !== progress.best?.delivery_id && (
            <Link
              href={`/deliveries/${progress.worst.delivery_id}`}
              className="lift block border-t border-border pt-3"
            >
              <span className="kicker">Most to work on</span>
              <span className="mt-1 block font-display text-xl">
                Technique {progress.worst.technique.toFixed(0)} →
              </span>
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
