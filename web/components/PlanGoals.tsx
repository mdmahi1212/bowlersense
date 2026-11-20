"use client";

import { useState, type FormEvent } from "react";
import { api, errorText } from "@/lib/api";
import type { DrillPlan, Goal, ProgressOut } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { cn, fmtDate } from "@/lib/utils";
import { Button } from "./ui/Button";

const METRIC_LABEL: Record<Goal["metric"], [string, string]> = {
  technique: ["Technique", "/100"],
  injury: ["Injury load", "/100"],
  elbow_extension_max: ["Elbow straightening", "°"],
  release_height_pct: ["Release height", "% ht"],
  stride_pct: ["Stride", "% ht"],
  arm_speed_deg_s: ["Arm speed", "°/s"],
};

/** A multi-week drill plan built from the bowler's recurring faults, ticked off as they go. */
export function DrillPlanPanel({ bowlerId }: { bowlerId: string }) {
  const plan = useAsync(() => api.plan(bowlerId), [bowlerId]);
  const [local, setLocal] = useState<DrillPlan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const p = local ?? plan.data ?? null;

  async function build() {
    setBusy(true);
    setError(null);
    try {
      setLocal(await api.makePlan(bowlerId));
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }
  function tick(si: number, ii: number) {
    if (!p) return;
    const next: DrillPlan = {
      ...p,
      sessions: p.sessions.map((s, a) =>
        a !== si
          ? s
          : { ...s, items: s.items.map((it, b) => (b === ii ? { ...it, done: !it.done } : it)) },
      ),
    };
    setLocal(next);
    api.savePlan(bowlerId, next).catch((err) => setError(errorText(err)));
  }

  if (plan.loading && !p) return <p className="text-muted">Loading the plan…</p>;
  if (!p)
    return (
      <div className="flex flex-wrap items-center gap-6">
        <p className="max-w-md text-muted">
          Build a practice plan from the faults that keep coming up in this bowler&apos;s recent
          sessions.
        </p>
        <Button onClick={build} disabled={busy}>
          {busy ? "Building…" : "Build a 4-week plan"}
        </Button>
        {error && <p className="text-sm text-bad">{error}</p>}
      </div>
    );
  const all = p.sessions.flatMap((s) => s.items);
  const done = all.filter((i) => i.done).length;
  const weeks = [...new Set(p.sessions.map((s) => s.week))];
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm text-muted">
            Built {fmtDate(p.created)} for: <b className="text-foreground">{p.faults.join(", ")}</b>
          </p>
          <div className="mt-2 flex items-center gap-3">
            <div className="h-1.5 w-56 bg-surface-2">
              <div
                className="h-full bg-good transition-all"
                style={{ width: `${(100 * done) / Math.max(1, all.length)}%` }}
              />
            </div>
            <span className="tabular text-sm">
              {done}/{all.length} drills done
            </span>
          </div>
        </div>
        <Button variant="secondary" onClick={build} disabled={busy}>
          {busy ? "Rebuilding…" : "Rebuild from latest faults"}
        </Button>
      </div>
      {error && <p className="text-sm text-bad">{error}</p>}
      <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
        {weeks.map((w) => (
          <div
            key={w}
            data-reveal
            style={{ "--d": `${w * 70}ms` } as React.CSSProperties}
            className="border-t border-foreground pt-3"
          >
            <p className="font-display text-xl">Week {w}</p>
            {p.sessions.map((s, si) =>
              s.week !== w ? null : (
                <div key={si} className="mt-3">
                  <p className="kicker">Session {s.session}</p>
                  <ul className="mt-1 space-y-1.5">
                    {s.items.map((it, ii) => (
                      <li key={ii}>
                        <label className="flex cursor-pointer gap-2 text-sm" title={it.description}>
                          <input
                            type="checkbox"
                            checked={it.done}
                            onChange={() => tick(si, ii)}
                            className="mt-1"
                          />
                          <span className={cn(it.done && "text-muted line-through")}>
                            {it.drill} <span className="text-xs text-muted">· {it.reps}</span>
                          </span>
                        </label>
                      </li>
                    ))}
                  </ul>
                </div>
              ),
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Targets on progress measures, checked against every session. */
export function GoalsPanel({
  bowlerId,
  progress,
}: {
  bowlerId: string;
  progress: ProgressOut | null;
}) {
  const saved = useAsync(() => api.goals(bowlerId), [bowlerId]);
  const [local, setLocal] = useState<Goal[] | null>(null);
  const goals = local ?? saved.data?.goals ?? [];
  const [metric, setMetric] = useState<Goal["metric"]>("technique");
  const [op, setOp] = useState<Goal["op"]>("above");
  const [target, setTarget] = useState("75");
  const [error, setError] = useState<string | null>(null);

  function save(next: Goal[]) {
    setLocal(next);
    api.saveGoals(bowlerId, next).catch((err) => setError(errorText(err)));
  }
  function add(e: FormEvent) {
    e.preventDefault();
    const t = Number(target);
    if (!Number.isFinite(t)) return setError("The target must be a number.");
    setError(null);
    save([...goals, { metric, op, target: t, label: "" }]);
  }
  const pts = progress?.points ?? [];
  const input = "border-b border-foreground/60 bg-transparent py-1.5 outline-none";

  return (
    <div className="space-y-6">
      {goals.length === 0 && (
        <p className="text-muted">No goals yet. Set one below, e.g. technique above 75.</p>
      )}
      <ul className="divide-y divide-border border-y border-border">
        {goals.map((g, i) => {
          const [label, unit] = METRIC_LABEL[g.metric];
          const hits = pts.map((p) => {
            const v = p[g.metric] as number | null;
            return v === null ? null : g.op === "above" ? v >= g.target : v <= g.target;
          });
          const lastV = pts.length ? (pts[pts.length - 1][g.metric] as number | null) : null;
          const met = hits.length ? hits[hits.length - 1] : null;
          return (
            <li
              key={i}
              className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-6 py-3"
            >
              <div>
                <p className="font-display text-xl">
                  {label} {g.op} {g.target}
                  {unit}
                </p>
                <div className="mt-1.5 flex gap-1" aria-label="Hit or miss per session">
                  {hits.map((h, j) => (
                    <span
                      key={j}
                      title={fmtDate(pts[j].day)}
                      className={cn(
                        "h-2.5 w-2.5 rounded-full",
                        h === null ? "bg-surface-2" : h ? "bg-good" : "bg-bad/70",
                      )}
                    />
                  ))}
                </div>
              </div>
              <span
                className={cn(
                  "text-sm",
                  met ? "text-good" : met === false ? "text-bad" : "text-muted",
                )}
              >
                {met === null ? "no data" : met ? "Met" : `Not yet (${lastV?.toFixed(1)}${unit})`}
              </span>
              <button
                type="button"
                onClick={() => save(goals.filter((_, j) => j !== i))}
                className="text-sm text-muted hover:text-bad"
              >
                Remove
              </button>
            </li>
          );
        })}
      </ul>
      <form onSubmit={add} className="flex flex-wrap items-end gap-4 text-sm">
        <label className="flex flex-col">
          <span className="kicker">Measure</span>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as Goal["metric"])}
            className={input}
          >
            {Object.entries(METRIC_LABEL).map(([k, [l]]) => (
              <option key={k} value={k}>
                {l}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col">
          <span className="kicker">Should be</span>
          <select
            value={op}
            onChange={(e) => setOp(e.target.value as Goal["op"])}
            className={input}
          >
            <option value="above">at or above</option>
            <option value="below">at or below</option>
          </select>
        </label>
        <label className="flex flex-col">
          <span className="kicker">Target</span>
          <input
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            inputMode="decimal"
            className={cn(input, "w-24")}
          />
        </label>
        <Button type="submit">Add goal</Button>
        {error && <span className="text-bad">{error}</span>}
      </form>
    </div>
  );
}
