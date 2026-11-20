"use client";

import Link from "next/link";
import { use, useState } from "react";
import { CalibrationPanel } from "@/components/CalibrationPanel";
import { CoachChat } from "@/components/CoachChat";
import { FatigueTrend } from "@/components/FatigueTrend";
import { VerdictWord, sessionVerdict } from "@/components/LegalityVerdict";
import { PersonPicker } from "@/components/PersonPicker";
import { PitchMap } from "@/components/PitchMap";
import { RiskGauge } from "@/components/RiskGauge";
import { StatusBadge } from "@/components/StatusBadge";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { Empty, ErrorBox, Loading } from "@/components/ui/States";
import { api, type SessionEvent } from "@/lib/api";
import type { SessionOut, SessionResult } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { useSessionStatus } from "@/lib/useSessionStatus";
import { bandOf, bandStyle, fmt } from "@/lib/utils";

function eventOf(s: SessionOut): SessionEvent {
  return {
    status: s.session.status,
    progress: s.job?.progress ?? 0,
    stage: s.job?.stage ?? null,
    attempts: s.job?.attempts ?? 0,
    error: s.session.error,
  };
}

export default function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const session = useAsync(() => api.session(id), [id]);
  if (session.error) {
    return (
      <ErrorBox
        error={session.error}
        action={
          <Link href="/sessions" className="text-accent underline">
            Back to sessions
          </Link>
        }
      />
    );
  }
  if (!session.data) return <Loading label="Loading session…" />;
  return (
    <SessionView key={session.data.session.status} out={session.data} reload={session.reload} />
  );
}

function SessionView({ out, reload }: { out: SessionOut; reload: () => void }) {
  const id = out.session.id;
  const live = useSessionStatus(id, eventOf(out), reload); // fetch the results once finished
  const status = live?.status ?? out.session.status;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="font-display text-[10px] uppercase tracking-[0.2em] text-muted">
            <Link href="/sessions" className="hover:text-foreground">
              Sessions
            </Link>{" "}
            / {id.slice(0, 8)}
            {out.result &&
              ` · ${out.result.view} view · ${out.result.bowling_arm}-arm · ${out.result.pose_backend}`}
          </p>
          <h1 className="font-display text-2xl font-semibold tracking-wide">
            {out.session.original_name ?? "Session"}
          </h1>
        </div>
        {status !== "deleted" && <StatusBadge status={status} />}
      </div>
      {(status === "queued" || status === "processing") && <Progress event={live} />}
      {status === "failed" && (
        <ErrorBox
          error={(live?.error ?? out.session.error ?? "The analysis failed.").split("\n")[0]}
          action={
            <Link href="/" className="text-accent underline">
              Try another video
            </Link>
          }
        />
      )}
      {status === "failed" && <WrongPerson sessionId={id} reload={reload} failed />}
      {status === "deleted" && <Empty title="This session was deleted." />}
      {status === "done" && out.result && out.result.pose_backend !== "pose-json" && (
        <WrongPerson sessionId={id} reload={reload} />
      )}
      {status === "done" && out.result && <Results out={out} result={out.result} reload={reload} />}
      {status === "done" && !out.result && <Loading label="Loading results…" />}
    </div>
  );
}

function WrongPerson({
  sessionId,
  reload,
  failed = false,
}: {
  sessionId: string;
  reload: () => void;
  failed?: boolean;
}) {
  const [open, setOpen] = useState(failed);
  return (
    <Card>
      <CardTitle
        aside={
          <Button variant="ghost" onClick={() => setOpen((o) => !o)}>
            {open ? "Close" : "Wrong person? Pick the bowler"}
          </Button>
        }
      >
        Bowler
      </CardTitle>
      {open ? (
        <PersonPicker sessionId={sessionId} onChosen={reload} />
      ) : (
        <p className="text-sm text-muted">
          If the skeleton follows someone other than the bowler, pick the right person and the video
          is analysed again.
        </p>
      )}
    </Card>
  );
}

function Progress({ event }: { event: SessionEvent | null }) {
  const pct = Math.round(event?.progress ?? 0);
  return (
    <Card aria-live="polite">
      <div className="mb-2 flex justify-between text-sm">
        <span>
          {event?.status === "queued"
            ? "Waiting for the worker…"
            : `Analysing: ${event?.stage ?? "starting"}`}
          {event && event.attempts > 1 ? ` (attempt ${event.attempts})` : ""}
        </span>
        <span>{pct}%</span>
      </div>
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        className="h-2 overflow-hidden rounded-full bg-surface-2"
      >
        <div className="h-full bg-accent transition-all" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-2 text-xs text-muted">
        Pose estimation on a CPU can take a few minutes for a long video. You can leave this page.
      </p>
    </Card>
  );
}

function Results({
  out,
  result,
  reload,
}: {
  out: SessionOut;
  result: SessionResult;
  reload: () => void;
}) {
  const [calibrating, setCalibrating] = useState(false);
  const ds = result.deliveries;
  const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
  const technique = avg(ds.map((d) => d.technique_score));
  const injury = avg(ds.map((d) => d.injury.overall));
  const flagged = ds.filter((d) => d.legality.flagged).length;
  const screened = ds.filter((d) => d.legality.screened);
  const maxExt = screened.length
    ? Math.max(...screened.map((d) => d.legality.elbow_extension_deg))
    : null;
  const speeds = ds.map((d) => d.speed_kmh).filter((v): v is number => v !== null);
  const band = bandStyle[bandOf(injury)];
  const faultCounts = new Map<string, number>();
  ds.forEach((d) =>
    d.faults.forEach((f) => faultCounts.set(f.name, (faultCounts.get(f.name) ?? 0) + 1)),
  );
  const topFaults = [...faultCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);

  return (
    <>
      {result.quality_warnings.length > 0 && (
        <div className="rounded-sm border border-warn/40 bg-warn/10 px-4 py-2.5 text-sm">
          <span className="mr-2 font-display text-xs font-semibold uppercase tracking-wider text-warn">
            Recording quality
          </span>
          {result.quality_warnings.join(" · ")}
        </div>
      )}

      <div className="grid grid-cols-2 divide-border overflow-hidden rounded-md border border-border bg-surface/70 sm:grid-cols-3 lg:grid-cols-6 lg:divide-x">
        <Kpi label="Technique" sub={`${ds.length} deliver${ds.length === 1 ? "y" : "ies"}`}>
          <span data-testid="technique-score">{technique.toFixed(0)}</span>
          <small>/100</small>
        </Kpi>
        <Kpi label="Injury load" sub={band.label}>
          <span className={band.text}>{injury.toFixed(0)}</span>
          <small>/100</small>
        </Kpi>
        <Kpi
          label="Legality"
          sub={
            screened.length
              ? `${screened.length - flagged} legal · ${flagged} not · max ${fmt(maxExt, 1, "°")}`
              : "not screened"
          }
        >
          <VerdictWord verdict={sessionVerdict(ds.map((d) => d.legality.verdict))} size="sm" />
        </Kpi>
        <Kpi
          label="Consistency"
          sub={result.consistency_score === null ? "needs 2+ balls" : "across session"}
        >
          {fmt(result.consistency_score)}
          {result.consistency_score !== null && <small>/100</small>}
        </Kpi>
        <Kpi label="Fatigue" sub={result.fatigue.alert ? "alert" : "no drift alert"}>
          <span className={result.fatigue.alert ? "text-warn" : undefined}>
            {result.fatigue.fatigue_index.toFixed(0)}
          </span>
          <small>/100</small>
        </Kpi>
        <Kpi label="Top speed" sub={speeds.length ? "calibrated pitch" : "calibrate the pitch"}>
          {speeds.length ? Math.max(...speeds).toFixed(0) : "—"}
          {speeds.length > 0 && <small> km/h</small>}
        </Kpi>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Panel title="Deliveries" hint="Open one for the frame-by-frame workspace">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-[10px] uppercase tracking-wider text-muted">
                <tr className="border-b border-border">
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Technique</th>
                  <th className="px-3 py-2 font-medium">Load</th>
                  <th className="px-3 py-2 font-medium">Legality</th>
                  <th className="px-3 py-2 font-medium">Release ht</th>
                  <th className="px-3 py-2 font-medium">Stride</th>
                  <th className="px-3 py-2 font-medium">Faults</th>
                  <th className="px-3 py-2 font-medium">Speed · length</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {ds.map((d, i) => {
                  const b = bandStyle[bandOf(d.injury.overall)];
                  return (
                    <tr
                      key={d.delivery_idx}
                      className="border-b border-border/60 hover:bg-surface-2/60"
                    >
                      <td className="px-3 py-2 font-display font-semibold">
                        {String(d.delivery_idx + 1).padStart(2, "0")}
                      </td>
                      <td className="px-3 py-2">
                        <Bar value={d.technique_score} />
                      </td>
                      <td className={`tabular px-3 py-2 ${b.text}`}>
                        {d.injury.overall.toFixed(0)} <span className="text-[11px]">{b.label}</span>
                      </td>
                      <td className="px-3 py-2">
                        <VerdictWord verdict={d.legality.verdict} size="sm" />
                        {d.legality.screened && (
                          <span className="tabular ml-1 text-muted">
                            {fmt(d.legality.elbow_extension_deg, 1, "°")}
                          </span>
                        )}
                      </td>
                      <td className="tabular px-3 py-2">{fmt(d.release_height_pct, 0, "%")}</td>
                      <td className="tabular px-3 py-2">{fmt(d.stride_pct, 0, "%")}</td>
                      <td className="px-3 py-2">
                        {d.faults.length ? (
                          <span
                            title={d.faults.map((f) => f.name).join(", ")}
                            className="text-warn"
                          >
                            {d.faults.length} · {d.faults[0].name}
                          </span>
                        ) : (
                          <span className="text-muted">None</span>
                        )}
                      </td>
                      <td className="tabular px-3 py-2">
                        {fmt(d.speed_kmh, 0, " km/h")} · {d.length ?? "—"}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {out.delivery_ids[i] && (
                          <Link
                            href={`/deliveries/${out.delivery_ids[i]}`}
                            className="rounded-sm border border-accent/50 px-2 py-0.5 text-xs text-accent hover:bg-accent hover:text-background"
                          >
                            Details
                          </Link>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>

        <div className="space-y-4">
          <Panel title="Injury load">
            <div className="flex justify-center">
              <RiskGauge score={injury} />
            </div>
          </Panel>
          <Panel title="Recurring faults" hint="How many deliveries show each">
            {topFaults.length === 0 ? (
              <p className="text-sm text-muted">No technique faults found.</p>
            ) : (
              <ul className="space-y-2 text-sm">
                {topFaults.map(([name, n]) => (
                  <li key={name}>
                    <div className="flex justify-between">
                      <span>{name}</span>
                      <span className="tabular text-muted">
                        {n}/{ds.length}
                      </span>
                    </div>
                    <div className="mt-1 h-1 bg-surface-2">
                      <div
                        className="h-full bg-warn"
                        style={{ width: `${(100 * n) / ds.length}%` }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Panel
          title="Fatigue drift"
          hint={
            result.workload.acwr !== null
              ? `Workload ACWR ${result.workload.acwr.toFixed(2)}${result.workload.flagged ? " (high)" : ""}`
              : result.workload.note
          }
          aside={
            result.fatigue.alert ? (
              <Badge className="border-warn/40 bg-warn/15 text-warn">Alert</Badge>
            ) : undefined
          }
        >
          <FatigueTrend fatigue={result.fatigue} />
          {result.fatigue.alert_reasons.length > 0 && (
            <p className="mt-2 text-sm text-warn">{result.fatigue.alert_reasons.join(" ")}</p>
          )}
        </Panel>
        <Panel
          title="Pitch map"
          aside={
            <Button variant="ghost" onClick={() => setCalibrating((c) => !c)}>
              {calibrating ? "Close" : "Calibrate pitch"}
            </Button>
          }
        >
          {calibrating ? (
            <CalibrationPanel
              sessionId={out.session.id}
              initial={out.session.calibration}
              onSaved={() => {
                setCalibrating(false);
                reload();
              }}
            />
          ) : (
            <PitchMap deliveries={ds} />
          )}
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel
          title="Coach report"
          hint={
            result.coach_source === "ollama"
              ? "Written by your local LLM coach"
              : "Rule-based report (local LLM not used)"
          }
          aside={
            <a
              href={api.reportUrl(out.session.id)}
              className="rounded-sm border border-border px-2.5 py-1 text-xs hover:border-accent"
              download
            >
              Download PDF
            </a>
          }
        >
          <div className="whitespace-pre-wrap text-sm leading-relaxed" data-testid="coach-report">
            {result.coach_report}
          </div>
        </Panel>
        <Panel title="Ask the coach" hint="Answers use this session's numbers">
          <CoachChat sessionId={out.session.id} />
        </Panel>
      </div>
      {result.similar_pros.length > 0 && (
        <Panel
          title="Closest reference actions"
          aside={
            <Link href="/compare" className="text-xs text-accent hover:underline">
              Compare side by side
            </Link>
          }
        >
          <ul className="grid gap-3 sm:grid-cols-3">
            {result.similar_pros.map(([name, sim]) => (
              <li key={name} className="text-sm">
                <div className="flex justify-between">
                  <span>{name}</span>
                  <span className="tabular text-muted">{sim.toFixed(0)}%</span>
                </div>
                <div className="mt-1 h-1 bg-surface-2">
                  <div className="h-full bg-accent" style={{ width: `${sim}%` }} />
                </div>
              </li>
            ))}
          </ul>
        </Panel>
      )}
      <p className="text-[11px] text-muted">
        Legality is 2D video screening, not the official ICC test (which uses 3D motion capture).
        Injury load is a rule-based estimate, not medical advice.
      </p>
    </>
  );
}

function Kpi({ label, sub, children }: { label: string; sub: string; children: React.ReactNode }) {
  return (
    <div className="border-b border-border px-4 py-3 lg:border-b-0">
      <p className="font-display text-[10px] uppercase tracking-[0.16em] text-muted">{label}</p>
      <p className="tabular mt-1 font-display text-3xl font-semibold leading-none [&_small]:ml-0.5 [&_small]:text-xs [&_small]:text-muted">
        {children}
      </p>
      <p className="mt-1.5 truncate text-[11px] text-muted">{sub}</p>
    </div>
  );
}

function Bar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2">
      <span className="tabular w-6">{value.toFixed(0)}</span>
      <div className="h-1.5 w-20 bg-surface-2">
        <div
          className="h-full bg-accent"
          style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
        />
      </div>
    </div>
  );
}

function Panel({
  title,
  hint,
  aside,
  children,
}: {
  title: string;
  hint?: string;
  aside?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-md border border-border bg-surface/70">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <div>
          <h2 className="font-display text-sm font-semibold uppercase tracking-[0.14em]">
            {title}
          </h2>
          {hint && <p className="text-[11px] text-muted">{hint}</p>}
        </div>
        {aside}
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}
