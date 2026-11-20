"use client";

import Link from "next/link";
import { use, useState } from "react";
import { CalibrationPanel } from "@/components/CalibrationPanel";
import { CoachChat } from "@/components/CoachChat";
import { FatigueTrend } from "@/components/FatigueTrend";
import { VerdictWord, sessionVerdict } from "@/components/LegalityVerdict";
import { PersonPicker } from "@/components/PersonPicker";
import { PitchMap } from "@/components/PitchMap";
import { StatusBadge } from "@/components/StatusBadge";
import { Card } from "@/components/ui/Card";
import { Empty, ErrorBox, Loading } from "@/components/ui/States";
import { api, type SessionEvent } from "@/lib/api";
import type { SessionOut, SessionResult } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { useSessionStatus } from "@/lib/useSessionStatus";
import { PageHead } from "@/components/ui/PageHead";
import { SectionHead } from "@/components/ui/SectionHead";
import { bandOf, bandStyle, cleanName, cn, fmt } from "@/lib/utils";

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
  const [picking, setPicking] = useState(status === "failed");
  const r = out.result;

  return (
    <div className="space-y-14">
      <PageHead
        kicker={
          <>
            <Link href="/sessions" className="hover:text-foreground">
              Sessions
            </Link>{" "}
            / Session report
          </>
        }
        title={cleanName(out.session.original_name)}
        deck={
          r
            ? `${r.deliveries.length} deliver${r.deliveries.length === 1 ? "y" : "ies"}, ${r.bowling_arm}-arm bowler, filmed ${r.view}-on.`
            : `Session ${id.slice(0, 8)}`
        }
        aside={
          <>
            {status !== "deleted" && status !== "done" && <StatusBadge status={status} />}
            {(status === "done" || status === "failed") && (
              <button
                type="button"
                onClick={() => setPicking((o) => !o)}
                className="text-muted underline-offset-4 hover:text-foreground hover:underline"
              >
                {picking ? "Close bowler picker" : "Wrong bowler?"}
              </button>
            )}
            {status === "done" && (
              <a
                href={api.reportUrl(id)}
                download
                className="bg-foreground px-4 py-2 text-background hover:bg-accent"
              >
                Download PDF report
              </a>
            )}
          </>
        }
      />

      {picking && (
        <Card>
          <p className="mb-3 text-sm text-muted">
            If the skeleton follows someone other than the bowler, pick the right person and the
            video is analysed again.
          </p>
          <PersonPicker sessionId={id} onChosen={reload} />
        </Card>
      )}
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
      {status === "done" && r && <Results out={out} result={r} reload={reload} />}
      {status === "done" && !r && <Loading label="Loading results…" />}
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
        className="h-1.5 overflow-hidden rounded-full bg-surface-2"
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
  const allVerified = ds.every((d) => d.legality.screened);
  const maxExt = Math.max(0, ...ds.map((d) => d.legality.elbow_extension_deg));
  const limit = ds[0]?.legality.threshold_deg ?? 15;
  const reason = ds.find((d) => !d.legality.screened)?.legality.not_screened_reason;
  const band = bandStyle[bandOf(injury)];

  return (
    <>
      {/* verdict + headline numbers */}
      <section className="grid gap-12 border-t border-foreground pt-8 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <div>
          <p className="kicker">Legality screening</p>
          <div className="mt-4">
            <VerdictWord
              verdict={sessionVerdict(ds.map((d) => d.legality.verdict))}
              verified={allVerified}
            />
          </div>
          <p className="mt-5 max-w-xl font-display text-xl leading-snug">
            The elbow straightened at most{" "}
            <span className="tabular text-accent">{fmt(maxExt, 1, "°")}</span> between
            arm-horizontal and release, against a {limit.toFixed(0)}° limit
            {ds.length > 1 && ` (${ds.length - flagged} legal, ${flagged} not legal)`}.
          </p>
          {!allVerified && reason && (
            <p className="mt-4 max-w-xl border-l-2 border-warn pl-4 text-sm leading-relaxed text-muted">
              <span className="font-medium text-warn">Why it isn&apos;t certain. </span>
              {reason.replace(/^Unverified:\s*/, "")} The verdict is read straight from the angle,
              so treat it as a guide and re-film side-on at 120+ fps to confirm.
            </p>
          )}
          <p className="mt-4 text-xs text-muted">
            2D video screening, not the official ICC test (which uses 3D motion capture).
          </p>
        </div>
        <dl className="grid grid-cols-2 self-start border-l border-border">
          <Stat
            label="Technique"
            value={technique.toFixed(0)}
            unit="/100"
            testId="technique-score"
          />
          <Stat
            label="Injury load"
            value={injury.toFixed(0)}
            unit="/100"
            note={band.label}
            noteCls={band.text}
          />
          <Stat
            label="Consistency"
            value={fmt(result.consistency_score)}
            unit={result.consistency_score === null ? "" : "/100"}
            note={result.consistency_score === null ? "needs 2+ deliveries" : undefined}
          />
          <Stat
            label="Fatigue"
            value={result.fatigue.fatigue_index.toFixed(0)}
            unit="/100"
            note={result.fatigue.alert ? "drift alert" : "no drift"}
            noteCls={result.fatigue.alert ? "text-warn" : undefined}
          />
        </dl>
      </section>

      {/* deliveries */}
      <section>
        <SectionHead n="01" title="Deliveries" hint="Open one to step through it frame by frame" />
        <ul className="divide-y divide-border border-b border-border">
          {ds.map((d, i) => {
            const b = bandStyle[bandOf(d.injury.overall)];
            const href = out.delivery_ids[i] ? `/deliveries/${out.delivery_ids[i]}` : null;
            return (
              <li
                key={d.delivery_idx}
                className="grid grid-cols-[2.5rem_minmax(0,1fr)_auto] items-center gap-4 py-4 sm:grid-cols-[2.5rem_9rem_7rem_minmax(0,1fr)_auto]"
              >
                <span className="font-display text-3xl leading-none">{d.delivery_idx + 1}</span>
                <span>
                  <VerdictWord
                    verdict={d.legality.verdict}
                    size="sm"
                    verified={d.legality.screened}
                  />
                  <span className="tabular ml-2 text-sm">
                    {fmt(d.legality.elbow_extension_deg, 1, "°")}
                  </span>
                </span>
                <span className="hidden text-sm sm:block">
                  Technique <b className="tabular">{d.technique_score.toFixed(0)}</b>
                </span>
                <span className="hidden min-w-0 truncate text-sm text-muted sm:block">
                  <span className={b.text}>{b.label} load</span>
                  {d.faults.length > 0 && ` · ${d.faults.map((f) => f.name).join(", ")}`}
                  {d.speed_kmh !== null && ` · ${fmt(d.speed_kmh, 0, " km/h")}`}
                </span>
                {href && (
                  <Link
                    href={href}
                    className="justify-self-end border border-foreground px-3 py-1 text-sm hover:bg-foreground hover:text-background"
                  >
                    Details
                  </Link>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      {/* coach */}
      <section className="grid gap-10 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div>
          <SectionHead
            n="02"
            title="Coach report"
            hint={
              result.coach_source === "ollama"
                ? "Written by your local LLM coach"
                : "Rule-based report (local LLM not used)"
            }
          />
          <div
            className="max-w-[68ch] whitespace-pre-wrap font-display text-[17px] leading-8"
            data-testid="coach-report"
          >
            {result.coach_report.replace(/\*\*/g, "")}
          </div>
        </div>
        <div>
          <SectionHead title="Ask the coach" hint="Answers use this session's numbers" />
          <CoachChat sessionId={out.session.id} />
        </div>
      </section>

      {/* trends */}
      <section className="grid gap-10 lg:grid-cols-2">
        <div>
          <SectionHead
            n="03"
            title="Fatigue drift"
            hint={
              result.workload.acwr !== null
                ? `Workload ACWR ${result.workload.acwr.toFixed(2)}${result.workload.flagged ? " (high)" : ""}`
                : result.workload.note
            }
          />
          <FatigueTrend fatigue={result.fatigue} />
          {result.fatigue.alert_reasons.length > 0 && (
            <p className="mt-2 text-sm text-warn">{result.fatigue.alert_reasons.join(" ")}</p>
          )}
        </div>
        <div>
          <SectionHead
            title="Pitch map"
            hint="Ball speed, line and length need a calibrated pitch"
            aside={
              <button
                type="button"
                onClick={() => setCalibrating((c) => !c)}
                className="text-sm text-accent hover:underline"
              >
                {calibrating ? "Close" : "Calibrate pitch"}
              </button>
            }
          />
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
        </div>
      </section>

      {result.similar_pros.length > 0 && (
        <section>
          <SectionHead
            n="04"
            title="Closest reference actions"
            aside={
              <Link href="/compare" className="text-sm text-accent hover:underline">
                Compare side by side
              </Link>
            }
          />
          <ul className="grid gap-x-10 gap-y-4 sm:grid-cols-3">
            {result.similar_pros.map(([name, sim]) => (
              <li key={name} className="text-sm">
                <div className="flex justify-between">
                  <span>{name}</span>
                  <span className="tabular text-muted">{sim.toFixed(0)}%</span>
                </div>
                <div className="mt-2 h-px bg-border">
                  <div
                    className="h-0.5 -translate-y-px bg-foreground"
                    style={{ width: `${sim}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {result.quality_warnings.length > 0 && (
        <p className="text-sm text-muted">
          <span className="text-warn">Recording: </span>
          {result.quality_warnings.join(" ")}
        </p>
      )}
      <p className="text-xs text-muted">
        Injury load is a rule-based estimate, not medical advice.
      </p>
    </>
  );
}

function Stat({
  label,
  value,
  unit,
  note,
  noteCls,
  testId,
}: {
  label: string;
  value: string;
  unit: string;
  note?: string;
  noteCls?: string;
  testId?: string;
}) {
  return (
    <div className="border-b border-border px-6 py-5 [&:nth-last-child(-n+2)]:border-b-0">
      <dt className="kicker">{label}</dt>
      <dd className="mt-2 flex items-baseline gap-1">
        <span className="font-display text-6xl leading-none" data-testid={testId}>
          {value}
        </span>
        <span className="text-sm text-muted">{unit}</span>
      </dd>
      {note && <p className={cn("mt-2 text-sm", noteCls ?? "text-muted")}>{note}</p>}
    </div>
  );
}
