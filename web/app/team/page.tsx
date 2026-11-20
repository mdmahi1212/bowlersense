"use client";

import Link from "next/link";
import { CountUp } from "@/components/motion/CountUp";
import { PageHead } from "@/components/ui/PageHead";
import { Empty, ErrorBox, Loading } from "@/components/ui/States";
import { api } from "@/lib/api";
import type { BowlerRecord, ProgressOut, WorkloadOut } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { bandOf, bandStyle, cn, fmtDate } from "@/lib/utils";

interface Row {
  bowler: BowlerRecord;
  progress: ProgressOut | null;
  workload: WorkloadOut | null;
}

/** Every bowler at a glance: latest technique, load, legality and workload ratio. */
export default function TeamPage() {
  const team = useAsync(async (): Promise<Row[]> => {
    const bowlers = await api.bowlers();
    return Promise.all(
      bowlers.map(async (b) => ({
        bowler: b,
        progress: await api.progress(b.id).catch(() => null),
        workload: await api.workload(b.id).catch(() => null),
      })),
    );
  }, []);

  return (
    <div>
      <PageHead
        kicker="Players"
        title="Team view"
        deck="Every bowler on one sheet: their latest session, technique trend, load and legality. Sorted by who needs attention first."
      />
      {team.error && <ErrorBox error={team.error} />}
      {team.loading && !team.data && <Loading />}
      {team.data?.length === 0 && (
        <Empty title="No bowlers yet">
          <Link href="/bowlers" className="text-accent underline">
            Add a bowler profile
          </Link>{" "}
          to build the team sheet.
        </Empty>
      )}
      {team.data && team.data.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left">
            <thead>
              <tr className="kicker border-b border-foreground">
                <th className="py-3 font-semibold">Bowler</th>
                <th className="py-3 font-semibold">Last session</th>
                <th className="py-3 text-right font-semibold">Technique</th>
                <th className="py-3 text-right font-semibold">Trend</th>
                <th className="py-3 text-right font-semibold">Injury load</th>
                <th className="py-3 text-right font-semibold">Workload ratio</th>
                <th className="py-3 text-right font-semibold">Legality</th>
              </tr>
            </thead>
            <tbody>
              {[...team.data]
                .sort((a, b) => urgency(b) - urgency(a))
                .map((r, i) => {
                  const pts = r.progress?.points ?? [];
                  const last = pts[pts.length - 1];
                  const prev = pts[pts.length - 2];
                  const trend = last && prev ? last.technique - prev.technique : null;
                  const band = last ? bandStyle[bandOf(last.injury)] : null;
                  const acwr = r.workload?.latest?.acwr ?? null;
                  return (
                    <tr
                      key={r.bowler.id}
                      data-reveal
                      style={{ "--d": `${i * 60}ms` } as React.CSSProperties}
                      className="border-b border-border"
                    >
                      <td className="py-4">
                        <Link href="/bowlers" className="font-display text-2xl hover:text-accent">
                          {r.bowler.name}
                        </Link>
                        <span className="block text-xs text-muted">
                          {[
                            r.bowler.bowling_arm && `${r.bowler.bowling_arm}-arm`,
                            r.bowler.bowling_type,
                          ]
                            .filter(Boolean)
                            .join(" · ") || "—"}
                        </span>
                      </td>
                      <td className="py-4 text-sm text-muted">
                        {last ? (
                          <Link
                            href={`/sessions/${last.session_id}`}
                            className="hover:text-foreground"
                          >
                            {fmtDate(last.day)} · {last.deliveries} balls
                          </Link>
                        ) : (
                          "No sessions"
                        )}
                      </td>
                      <td className="py-4 text-right font-display text-3xl">
                        {last ? <CountUp value={last.technique} /> : "—"}
                      </td>
                      <td
                        className={cn(
                          "tabular py-4 text-right",
                          trend === null ? "text-muted" : trend >= 0 ? "text-good" : "text-bad",
                        )}
                      >
                        {trend === null
                          ? "—"
                          : `${trend >= 0 ? "▲" : "▼"} ${Math.abs(trend).toFixed(0)}`}
                      </td>
                      <td className={cn("py-4 text-right", band?.text)}>
                        {last ? `${last.injury.toFixed(0)} ${band?.label}` : "—"}
                      </td>
                      <td
                        className={cn(
                          "tabular py-4 text-right",
                          acwr !== null && (acwr > 1.3 || acwr < 0.8) ? "text-warn" : "",
                        )}
                      >
                        {acwr === null ? "—" : acwr.toFixed(2)}
                      </td>
                      <td className="py-4 text-right text-sm">
                        {last ? (
                          last.not_legal > 0 ? (
                            <span className="text-bad">{last.not_legal} not legal</span>
                          ) : (
                            <span className="text-good">All legal</span>
                          )
                        ) : (
                          "—"
                        )}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
          <p className="mt-4 text-xs text-muted">
            Workload ratio outside 0.8–1.3 is highlighted. Legality is 2D screening; injury load is
            a rule-based estimate, not medical advice.
          </p>
        </div>
      )}
    </div>
  );
}

/** Higher = needs attention sooner. */
function urgency(r: Row): number {
  const last = r.progress?.points.at(-1);
  const acwr = r.workload?.latest?.acwr ?? 1;
  return (last?.injury ?? 0) + (last?.not_legal ?? 0) * 30 + Math.abs(acwr - 1) * 40;
}
