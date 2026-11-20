"use client";

import { useState } from "react";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox, Loading } from "@/components/ui/States";
import { api } from "@/lib/api";
import type { ClassificationReport, RegressionReport } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/utils";

const pct = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`;
const num = (v: number | null | undefined, d = 3) =>
  v === null || v === undefined ? "—" : v.toFixed(d);

export default function EvaluationPage() {
  const report = useAsync(() => api.evaluation(), []);
  const [data, setData] = useState<"real" | "synthetic">("real");
  if (report.error)
    return (
      <div className="space-y-4">
        <h1 className="font-display text-4xl font-bold uppercase tracking-wide">
          Model <span className="text-accent">evaluation</span>
        </h1>
        <ErrorBox
          error={report.error}
          action={<code className="text-xs">python scripts/evaluate.py</code>}
        />
      </div>
    );
  if (!report.data) return <Loading label="Loading the evaluation…" />;
  const r = report.data;
  const cls = r.classification.filter((c) => c.data === data);
  const reg = r.regression.filter((c) => c.data === data);
  const scores = r.scores.filter((c) => c.data === data);
  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-end justify-between gap-3 border-b border-border pb-4">
        <div>
          <h1 className="font-display text-4xl font-bold uppercase tracking-wide">
            Model <span className="text-accent">evaluation</span>
          </h1>
          <p className="text-sm text-muted">
            Accuracy, precision, recall, F1 and confusion matrices per task. Generated{" "}
            {new Date(r.generated).toLocaleString()}.
          </p>
        </div>
        <div className="flex rounded-sm border border-border p-0.5" role="tablist">
          {(["real", "synthetic"] as const).map((k) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={data === k}
              onClick={() => setData(k)}
              className={cn(
                "px-4 py-1.5 font-display text-sm uppercase tracking-wider",
                data === k ? "bg-accent text-black" : "text-muted hover:text-foreground",
              )}
            >
              {k === "real" ? "Real footage" : "Synthetic ground truth"}
            </button>
          ))}
        </div>
      </header>
      <p className="text-sm text-muted">
        {data === "real"
          ? "Measured on the labelled match-footage dataset's held-out test videos (never used for training)."
          : "Measured on generated bowlers with exactly known answers: there are no real labels for legality, faults or joint angles."}
      </p>

      {cls.length > 0 && (
        <Card className="overflow-x-auto">
          <CardTitle>Classification summary</CardTitle>
          <table className="w-full text-sm">
            <thead className="text-left font-display text-[11px] uppercase tracking-wider text-muted">
              <tr>
                {[
                  "Task",
                  "n",
                  "Accuracy",
                  "Balanced",
                  "Macro P",
                  "Macro R",
                  "Macro F1",
                  "Weighted F1",
                  "Baseline",
                  "Coverage",
                ].map((h) => (
                  <th key={h} className="px-2 py-2">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tabular">
              {cls.map((c) => (
                <tr key={c.task} className="border-t border-border">
                  <td className="px-2 py-2 font-sans font-medium">{c.task}</td>
                  <td className="px-2">{c.n}</td>
                  <td className="px-2 font-semibold text-accent">{pct(c.accuracy)}</td>
                  <td className="px-2">{pct(c.balanced_accuracy)}</td>
                  <td className="px-2">{num(c.macro_precision)}</td>
                  <td className="px-2">{num(c.macro_recall)}</td>
                  <td className="px-2 font-semibold">{num(c.macro_f1)}</td>
                  <td className="px-2">{num(c.weighted_f1)}</td>
                  <td className="px-2 text-muted">{pct(c.baseline_accuracy)}</td>
                  <td className="px-2">{pct(c.coverage)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {cls.map((c) => (
          <ConfusionCard key={c.task} c={c} />
        ))}
      </div>

      {reg.length > 0 && (
        <Card className="overflow-x-auto">
          <CardTitle>Measurement accuracy (biomechanics)</CardTitle>
          <table className="w-full text-sm">
            <thead className="text-left font-display text-[11px] uppercase tracking-wider text-muted">
              <tr>
                {[
                  "Measurement",
                  "n",
                  "MAE",
                  "RMSE",
                  "R²",
                  "Bias",
                  "Max error",
                  "Unit",
                  "Measured vs true",
                ].map((h) => (
                  <th key={h} className="px-2 py-2">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tabular">
              {reg.map((m) => (
                <tr key={m.task} className="border-t border-border">
                  <td className="px-2 py-2 font-sans font-medium">{m.task}</td>
                  <td className="px-2">{m.n}</td>
                  <td className="px-2 font-semibold text-accent">{num(m.mae, 2)}</td>
                  <td className="px-2">{num(m.rmse, 2)}</td>
                  <td className="px-2">{num(m.r2, 3)}</td>
                  <td className="px-2">
                    {m.bias >= 0 ? "+" : ""}
                    {num(m.bias, 2)}
                  </td>
                  <td className="px-2">{num(m.max_abs_error, 2)}</td>
                  <td className="px-2 font-sans text-muted">{m.unit}</td>
                  <td className="px-2">
                    <Scatter m={m} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {scores.length > 0 && (
        <Card>
          <CardTitle>Other scores</CardTitle>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {scores.map((sc) => (
              <div
                key={sc.task + sc.metric}
                className="rounded-sm border border-border bg-surface-2 p-3"
              >
                <p className="font-display text-[11px] uppercase tracking-widest text-muted">
                  {sc.task}
                </p>
                <p className="text-xs text-muted">{sc.metric}</p>
                <p className="tabular mt-1 text-2xl font-semibold">
                  {sc.metric.startsWith("MAE") ? sc.value.toFixed(0) : pct(sc.value)}
                </p>
                <p className="text-[11px] text-muted">
                  n = {sc.n}
                  {sc.note ? ` · ${sc.note}` : ""}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}
      {r.notes.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-xs text-muted">
          {r.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ConfusionCard({ c }: { c: ClassificationReport }) {
  const max = Math.max(1, ...c.confusion.flat());
  return (
    <Card>
      <CardTitle
        aside={<span className="tabular text-xs text-accent">F1 {num(c.macro_f1, 2)}</span>}
      >
        {c.task}
      </CardTitle>
      <div className="overflow-x-auto">
        <table className="tabular mx-auto text-xs" aria-label={`${c.task} confusion matrix`}>
          <thead>
            <tr>
              <th className="p-1 text-[10px] font-normal text-muted">true ↓ / pred →</th>
              {c.labels.map((l) => (
                <th
                  key={l}
                  className="p-1 font-display text-[11px] uppercase tracking-wider text-muted"
                >
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {c.confusion.map((row, i) => (
              <tr key={c.labels[i]}>
                <th className="p-1 text-right font-display text-[11px] uppercase tracking-wider text-muted">
                  {c.labels[i]}
                </th>
                {row.map((v, j) => (
                  <td
                    key={j}
                    className={cn(
                      "h-12 w-16 border border-background text-center text-sm font-semibold",
                      i === j ? "text-black" : "text-foreground",
                    )}
                    style={{
                      background:
                        i === j
                          ? `rgba(198,244,50,${0.25 + 0.75 * (v / max)})`
                          : `rgba(255,77,77,${v ? 0.15 + 0.6 * (v / max) : 0.04})`,
                    }}
                  >
                    {v}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <table className="tabular mt-3 w-full text-xs">
        <thead className="text-muted">
          <tr>
            <th className="text-left font-normal">class</th>
            <th className="font-normal">precision</th>
            <th className="font-normal">recall</th>
            <th className="font-normal">F1</th>
            <th className="font-normal">n</th>
          </tr>
        </thead>
        <tbody>
          {c.per_class.map((p) => (
            <tr key={p.label} className="border-t border-border text-center">
              <td className="py-1 text-left font-sans">{p.label}</td>
              <td>{num(p.precision, 2)}</td>
              <td>{num(p.recall, 2)}</td>
              <td className="font-semibold">{num(p.f1, 2)}</td>
              <td className="text-muted">{p.support}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-muted">
        Accuracy {pct(c.accuracy)} (baseline {pct(c.baseline_accuracy)})
        {c.abstained > 0 ? ` · declined ${c.abstained} (coverage ${pct(c.coverage)})` : ""}.{" "}
        {c.description}
      </p>
    </Card>
  );
}

function Scatter({ m }: { m: RegressionReport }) {
  const all = [...m.truth, ...m.measured];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const sx = (v: number) => 4 + ((v - lo) / span) * 72;
  const sy = (v: number) => 40 - ((v - lo) / span) * 36;
  return (
    <svg
      viewBox="0 0 80 44"
      className="h-11 w-20"
      role="img"
      aria-label={`${m.task}: measured vs true`}
    >
      <line x1={sx(lo)} y1={sy(lo)} x2={sx(hi)} y2={sy(hi)} stroke="var(--border)" />
      {m.truth.map((t, i) => (
        <circle key={i} cx={sx(t)} cy={sy(m.measured[i])} r={1.4} fill="var(--accent)" />
      ))}
    </svg>
  );
}
