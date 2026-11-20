"use client";

import { api } from "@/lib/api";
import type { SessionOut } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/utils";

type Readiness = { sleep: number; soreness: number; effort: number };

/** The pre-session readiness check, the spell tracker (overs against junior fast-bowling
 *  limits) and the workload ratio, with one plain-language load warning. */
export function ReadinessSpell({ out, balls }: { out: SessionOut; balls: number }) {
  const bowlerId = out.session.bowler_id;
  const workload = useAsync(
    () => (bowlerId ? api.workload(bowlerId) : Promise.resolve(null)),
    [bowlerId],
  );
  const r = (out.session.options as { readiness?: Readiness | null } | null)?.readiness ?? null;
  const readiness = r ? Math.round(((r.sleep - 1) / 4) * 50 + ((5 - r.soreness) / 4) * 50) : null;
  const day = (out.session.recorded_at ?? out.session.created_at).slice(0, 10);
  const dayBalls =
    workload.data?.points.find((p) => String(p.day).slice(0, 10) === day)?.balls ?? balls;
  const limit = workload.data?.spell_limit ?? null;
  const overs = balls / 6;
  const dayOvers = dayBalls / 6;
  const acwr = workload.data?.latest?.acwr ?? null;
  const warnings: string[] = [];
  if (limit && overs > limit.overs_per_spell)
    warnings.push(
      `This session is ${overs.toFixed(1)} overs, over the ${limit.overs_per_spell}-over spell limit for an under-${limit.max_age + 1}.`,
    );
  if (limit && dayOvers > limit.overs_per_day)
    warnings.push(
      `${dayOvers.toFixed(1)} overs today is over the ${limit.overs_per_day}-over daily limit.`,
    );
  if (readiness !== null && readiness < 45 && r && r.effort >= 7)
    warnings.push("Low readiness with a high-effort plan: consider a lighter session.");
  if (acwr !== null && acwr > (workload.data?.flag_above ?? 1.5))
    warnings.push(
      `Workload ratio ${acwr.toFixed(2)} is high: load has jumped compared with recent weeks.`,
    );

  return (
    <div className="grid gap-10 md:grid-cols-3">
      <div data-reveal>
        <p className="kicker">Readiness</p>
        {r ? (
          <>
            <p
              className={cn(
                "mt-2 font-display text-5xl leading-none",
                readiness! < 45 ? "text-warn" : "",
              )}
            >
              {readiness}
              <span className="ml-1 text-base text-muted">/100</span>
            </p>
            <p className="mt-2 text-sm text-muted">
              Sleep {r.sleep}/5 · soreness {r.soreness}/5 · planned effort {r.effort}/10
            </p>
          </>
        ) : (
          <p className="mt-2 text-sm text-muted">
            No readiness check was filled in for this upload.
          </p>
        )}
      </div>
      <div data-reveal style={{ "--d": "100ms" } as React.CSSProperties}>
        <p className="kicker">Spell tracker</p>
        <p className="mt-2 font-display text-5xl leading-none">
          {Math.floor(balls / 6)}.{balls % 6}
          <span className="ml-1 text-base text-muted">overs</span>
        </p>
        {limit ? (
          <>
            <div className="mt-3 h-1.5 bg-surface-2">
              <div
                className={cn("h-full", overs > limit.overs_per_spell ? "bg-bad" : "bg-good")}
                style={{ width: `${Math.min(100, (overs / limit.overs_per_spell) * 100)}%` }}
              />
            </div>
            <p className="mt-2 text-sm text-muted">
              Age {limit.age}: up to {limit.overs_per_spell} overs a spell, {limit.overs_per_day} a
              day · {dayOvers.toFixed(1)} today
            </p>
          </>
        ) : (
          <p className="mt-2 text-sm text-muted">
            {bowlerId
              ? "No junior limit applies (add a date of birth for under-19s)."
              : "Link a bowler to track spells."}
          </p>
        )}
      </div>
      <div data-reveal style={{ "--d": "200ms" } as React.CSSProperties}>
        <p className="kicker">Load warning</p>
        {warnings.length === 0 ? (
          <p className="mt-2 font-display text-2xl text-good">Nothing to flag</p>
        ) : (
          <ul className="mt-2 space-y-2 text-sm">
            {warnings.map((w) => (
              <li key={w} className="border-l-2 border-warn pl-3">
                {w}
              </li>
            ))}
          </ul>
        )}
        <p className="mt-3 text-xs text-muted">
          Guidance only, not medical advice. Spell limits follow junior fast-bowling directives;
          check your board&apos;s current rules.
        </p>
      </div>
    </div>
  );
}
