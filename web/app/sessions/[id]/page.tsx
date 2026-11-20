"use client";

import Link from "next/link";
import { use, useState } from "react";
import { CalibrationPanel } from "@/components/CalibrationPanel";
import { CoachChat } from "@/components/CoachChat";
import { FatigueTrend } from "@/components/FatigueTrend";
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
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold">{out.session.original_name ?? "Session"}</h1>
          <p className="text-sm text-muted">Session {id.slice(0, 8)}</p>
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
      {status === "deleted" && <Empty title="This session was deleted." />}
      {status === "done" && out.result && <Results out={out} result={out.result} reload={reload} />}
      {status === "done" && !out.result && <Loading label="Loading results…" />}
    </div>
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
  const technique = ds.reduce((a, d) => a + d.technique_score, 0) / Math.max(1, ds.length);
  const injury = ds.reduce((a, d) => a + d.injury.overall, 0) / Math.max(1, ds.length);
  const flagged = ds.filter((d) => d.legality.flagged).length;
  const screened = ds.filter((d) => d.legality.screened);
  const maxExt = Math.max(
    ...ds.filter((d) => d.legality.screened).map((d) => d.legality.elbow_extension_deg),
  );

  return (
    <>
      {result.quality_warnings.length > 0 && (
        <Card className="border-warn/50 bg-warn/10">
          <p className="font-semibold text-warn">Recording quality</p>
          <ul className="mt-1 list-disc pl-5 text-sm">
            {result.quality_warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </Card>
      )}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardTitle>Technique</CardTitle>
          <p className="text-4xl font-bold" data-testid="technique-score">
            {technique.toFixed(0)}
            <span className="text-base text-muted">/100</span>
          </p>
          <p className="text-sm text-muted">
            {ds.length} deliver{ds.length === 1 ? "y" : "ies"} · {result.view} view ·{" "}
            {result.bowling_arm}-arm
          </p>
        </Card>
        <Card className="flex justify-center">
          <RiskGauge score={injury} />
        </Card>
        <Card>
          <CardTitle>Elbow legality</CardTitle>
          {screened.length === 0 ? (
            <Badge>Not screened</Badge>
          ) : flagged > 0 ? (
            <Badge className="border-bad/40 bg-bad/15 text-bad">
              Flagged: {flagged} of {screened.length}
            </Badge>
          ) : (
            <Badge className="border-good/40 bg-good/15 text-good">Within 15°</Badge>
          )}
          {screened.length > 0 && (
            <p className="mt-2 text-sm">Max extension {fmt(maxExt, 1, "°")}</p>
          )}
          <p className="mt-1 text-xs text-muted">{(screened[0] ?? ds[0])?.legality.note}</p>
        </Card>
        <Card>
          <CardTitle>Consistency</CardTitle>
          <p className="text-4xl font-bold">
            {fmt(result.consistency_score)}
            {result.consistency_score !== null && (
              <span className="text-base text-muted">/100</span>
            )}
          </p>
          <p className="text-sm text-muted">
            {result.consistency_score === null ? "Needs 2+ deliveries" : "Across this session"}
          </p>
        </Card>
      </div>

      <Card>
        <CardTitle>Deliveries</CardTitle>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-muted">
              <tr>
                <th className="p-2 font-medium">#</th>
                <th className="p-2 font-medium">Technique</th>
                <th className="p-2 font-medium">Injury risk</th>
                <th className="p-2 font-medium">Elbow</th>
                <th className="p-2 font-medium">Faults</th>
                <th className="p-2 font-medium">Speed</th>
                <th className="p-2 font-medium">Length</th>
                <th className="p-2" />
              </tr>
            </thead>
            <tbody>
              {ds.map((d, i) => {
                const band = bandStyle[bandOf(d.injury.overall)];
                return (
                  <tr key={d.delivery_idx} className="border-t border-border">
                    <td className="p-2">{d.delivery_idx + 1}</td>
                    <td className="p-2">{d.technique_score.toFixed(0)}</td>
                    <td className={`p-2 ${band.text}`}>
                      {d.injury.overall.toFixed(0)} ({band.label})
                    </td>
                    <td className={`p-2 ${d.legality.flagged ? "text-bad" : ""}`}>
                      {d.legality.screened ? (
                        <>
                          {fmt(d.legality.elbow_extension_deg, 1, "°")}
                          {d.legality.flagged ? " flagged" : ""}
                        </>
                      ) : (
                        <span className="text-muted">not screened</span>
                      )}
                    </td>
                    <td className="p-2">{d.faults.map((f) => f.name).join(", ") || "None"}</td>
                    <td className="p-2">{fmt(d.speed_kmh, 0, " km/h")}</td>
                    <td className="p-2">{d.length ?? "—"}</td>
                    <td className="p-2 text-right">
                      {out.delivery_ids[i] && (
                        <Link
                          href={`/deliveries/${out.delivery_ids[i]}`}
                          className="text-accent hover:underline"
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
      </Card>

      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <Card>
          <CardTitle>
            Fatigue trend{" "}
            {result.fatigue.alert && (
              <Badge className="ml-2 border-warn/40 bg-warn/15 text-warn">Alert</Badge>
            )}
          </CardTitle>
          <FatigueTrend fatigue={result.fatigue} />
          {result.fatigue.alert_reasons.length > 0 && (
            <p className="mt-2 text-sm text-warn">{result.fatigue.alert_reasons.join(" ")}</p>
          )}
          <p className="mt-2 text-xs text-muted">
            Fatigue index {result.fatigue.fatigue_index.toFixed(0)}/100 · workload:{" "}
            {result.workload.acwr !== null
              ? `ACWR ${result.workload.acwr.toFixed(2)}${result.workload.flagged ? " (high)" : ""}`
              : result.workload.note}
          </p>
        </Card>
        <Card>
          <CardTitle
            aside={
              <Button variant="ghost" onClick={() => setCalibrating((c) => !c)}>
                {calibrating ? "Close" : "Calibrate pitch"}
              </Button>
            }
          >
            Pitch map
          </CardTitle>
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
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardTitle
            aside={
              <a
                href={api.reportUrl(out.session.id)}
                className="text-sm text-accent hover:underline"
                download
              >
                Download PDF
              </a>
            }
          >
            Coach report
          </CardTitle>
          <div className="whitespace-pre-wrap text-sm leading-relaxed" data-testid="coach-report">
            {result.coach_report}
          </div>
          <p className="mt-3 text-xs text-muted">
            {result.coach_source === "ollama"
              ? "Written by your local LLM coach."
              : "Rule-based report (local LLM not used)."}
          </p>
        </Card>
        <Card>
          <CardTitle>Ask the coach</CardTitle>
          <CoachChat sessionId={out.session.id} />
        </Card>
      </div>
      {result.similar_pros.length > 0 && (
        <Card>
          <CardTitle>Closest reference actions</CardTitle>
          <ul className="space-y-1 text-sm">
            {result.similar_pros.map(([name, sim]) => (
              <li key={name} className="flex justify-between">
                <span>{name}</span>
                <span className="text-muted">{sim.toFixed(0)}% similar</span>
              </li>
            ))}
          </ul>
          <Link href="/compare" className="mt-2 inline-block text-sm text-accent hover:underline">
            Compare side by side
          </Link>
        </Card>
      )}
    </>
  );
}
