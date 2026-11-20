"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { PageHead } from "@/components/ui/PageHead";
import { ErrorBox, Loading } from "@/components/ui/States";
import { api } from "@/lib/api";
import type { SessionOut, SessionResult } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { cleanName, cn, fmtDate } from "@/lib/utils";

type Metric = [string, string, (r: SessionResult) => number | null, "up" | "down" | null];

const mean = (xs: (number | null | undefined)[]) => {
  const v = xs.filter((x): x is number => typeof x === "number" && Number.isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};
const bm = (r: SessionResult, key: string) =>
  mean(r.deliveries.map((d) => d.biomechanics.find((m) => m.key === key)?.value));

const METRICS: Metric[] = [
  ["Technique", "/100", (r) => mean(r.deliveries.map((d) => d.technique_score)), "up"],
  ["Injury load", "/100", (r) => mean(r.deliveries.map((d) => d.injury.overall)), "down"],
  [
    "Max elbow straightening",
    "°",
    (r) => Math.max(...r.deliveries.map((d) => d.legality.elbow_extension_deg)),
    "down",
  ],
  ["Release height", "% ht", (r) => mean(r.deliveries.map((d) => d.release_height_pct)), "up"],
  ["Stride", "% ht", (r) => mean(r.deliveries.map((d) => d.stride_pct)), null],
  ["Arm speed", "°/s", (r) => bm(r, "arm_angular_velocity"), "up"],
  ["Consistency", "/100", (r) => r.consistency_score, "up"],
  ["Fatigue", "/100", (r) => r.fatigue.fatigue_index, "down"],
  ["Faults per ball", "", (r) => mean(r.deliveries.map((d) => d.faults.length)), "down"],
];

export default function CompareSessionsPage() {
  return (
    <Suspense fallback={<Loading />}>
      <CompareSessions />
    </Suspense>
  );
}

function CompareSessions() {
  const q = useSearchParams();
  const router = useRouter();
  const a = q.get("a") ?? "";
  const b = q.get("b") ?? "";
  const list = useAsync(() => api.sessions(), []);
  const sa = useAsync(() => (a ? api.session(a) : Promise.resolve(null)), [a]);
  const sb = useAsync(() => (b ? api.session(b) : Promise.resolve(null)), [b]);
  const done = (list.data ?? []).filter((s) => s.status === "done");
  const pick = (k: "a" | "b", v: string) => {
    const p = new URLSearchParams(q.toString());
    p.set(k, v);
    router.replace(`/sessions/compare?${p.toString()}`);
  };
  const select = (k: "a" | "b", v: string) => (
    <select
      value={v}
      onChange={(e) => pick(k, e.target.value)}
      className="w-full border-b border-foreground/60 bg-transparent py-2 font-display text-xl outline-none"
    >
      <option value="">Choose a session…</option>
      {done.map((s) => (
        <option key={s.id} value={s.id}>
          {cleanName(s.original_name)} · {fmtDate(s.created_at)}
        </option>
      ))}
    </select>
  );
  const ra = sa.data?.result ?? null;
  const rb = sb.data?.result ?? null;

  return (
    <div className="space-y-10">
      <PageHead
        kicker="Insight"
        title="Session against session"
        deck="Before and after: put two whole sessions side by side, for example the start and end of a month of drills."
      />
      {(list.error || sa.error || sb.error) && (
        <ErrorBox error={(list.error ?? sa.error ?? sb.error)!} />
      )}
      <div className="grid gap-8 border-t border-foreground pt-6 md:grid-cols-2">
        <label className="block">
          <span className="kicker">Before (A)</span>
          {select("a", a)}
          <Meta s={sa.data} />
        </label>
        <label className="block">
          <span className="kicker">After (B)</span>
          {select("b", b)}
          <Meta s={sb.data} />
        </label>
      </div>
      {ra && rb ? (
        <table className="w-full">
          <thead>
            <tr className="kicker border-b border-foreground text-left">
              <th className="py-3 font-semibold">Measure</th>
              <th className="py-3 text-right font-semibold">A</th>
              <th className="py-3 text-right font-semibold">B</th>
              <th className="py-3 text-right font-semibold">Change</th>
            </tr>
          </thead>
          <tbody>
            {METRICS.map(([name, unit, fn, better], i) => {
              const va = fn(ra);
              const vb = fn(rb);
              const d = va !== null && vb !== null ? vb - va : null;
              const good =
                d === null || better === null || Math.abs(d) < 0.05
                  ? null
                  : better === "up"
                    ? d > 0
                    : d < 0;
              return (
                <tr
                  key={name}
                  data-reveal
                  style={{ "--d": `${i * 50}ms` } as React.CSSProperties}
                  className="border-b border-border"
                >
                  <td className="py-3.5">{name}</td>
                  <td className="tabular py-3.5 text-right">
                    {va === null ? "—" : `${va.toFixed(1)}${unit}`}
                  </td>
                  <td className="tabular py-3.5 text-right">
                    {vb === null ? "—" : `${vb.toFixed(1)}${unit}`}
                  </td>
                  <td
                    className={cn(
                      "tabular py-3.5 text-right font-medium",
                      good === null ? "text-muted" : good ? "text-good" : "text-bad",
                    )}
                  >
                    {d === null ? "—" : `${d > 0 ? "+" : ""}${d.toFixed(1)}${unit}`}
                    {good !== null && (
                      <span className="ml-2 text-xs">{good ? "better" : "worse"}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <p className="font-display text-2xl text-muted">Choose two finished sessions to compare.</p>
      )}
    </div>
  );
}

function Meta({ s }: { s: SessionOut | null | undefined }) {
  if (!s?.result) return null;
  return (
    <span className="mt-2 block text-sm text-muted">
      {s.result.deliveries.length} deliveries · {s.result.bowling_arm}-arm · {s.result.view} view
    </span>
  );
}
