"use client";

import { PageHead } from "@/components/ui/PageHead";
import Link from "next/link";
import { useState, type FormEvent } from "react";
import { StatusBadge } from "@/components/StatusBadge";
import { WorkloadChart } from "@/components/WorkloadChart";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { Empty, ErrorBox, Loading } from "@/components/ui/States";
import { api, errorText } from "@/lib/api";
import type { BowlerRecord } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { ProgressPanel } from "@/components/ProgressPanel";
import { SectionHead } from "@/components/ui/SectionHead";
import { fmtDate } from "@/lib/utils";

export default function BowlersPage() {
  const bowlers = useAsync(() => api.bowlers(), []);
  const [selected, setSelected] = useState<string>("");
  const current = bowlers.data?.find((b) => b.id === selected) ?? bowlers.data?.[0] ?? null;

  return (
    <div>
      <PageHead
        kicker="Players"
        title="Bowlers"
        deck="Profiles, their sessions and training load. Acute:chronic workload between 0.8 and 1.3 is the usual safe band."
      />
      <div className="grid gap-10 lg:grid-cols-[280px_minmax(0,1fr)]">
        <div className="space-y-4">
          <NewBowler
            onCreated={(b) => {
              bowlers.reload();
              setSelected(b.id);
            }}
          />
          {bowlers.loading && !bowlers.data && <Loading />}
          {bowlers.error && <ErrorBox error={bowlers.error} />}
          {bowlers.data?.length === 0 && <Empty title="No bowler profiles yet" />}
          {bowlers.data && bowlers.data.length > 0 && (
            <Card className="p-2">
              <ul>
                {bowlers.data.map((b) => (
                  <li key={b.id}>
                    <button
                      type="button"
                      onClick={() => setSelected(b.id)}
                      aria-current={current?.id === b.id}
                      className={`w-full px-3 py-2 text-left text-sm ${current?.id === b.id ? "bg-surface-2" : "hover:bg-surface-2"}`}
                    >
                      <span className="font-medium">{b.name}</span>
                      <span className="block text-xs text-muted">
                        {[
                          b.bowling_arm && `${b.bowling_arm}-arm`,
                          b.bowling_type,
                          b.height_cm && `${b.height_cm} cm`,
                        ]
                          .filter(Boolean)
                          .join(" · ") || "No details"}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
        {current && <BowlerDetail key={current.id} bowler={current} />}
      </div>
    </div>
  );
}

function BowlerDetail({ bowler }: { bowler: BowlerRecord }) {
  const workload = useAsync(() => api.workload(bowler.id), [bowler.id]);
  const sessions = useAsync(() => api.sessions(bowler.id), [bowler.id]);
  const progress = useAsync(() => api.progress(bowler.id), [bowler.id]);
  return (
    <div className="space-y-10">
      <section>
        <SectionHead
          n="01"
          title={`${bowler.name}: progress`}
          hint="Every finished session, oldest to newest. Pick a measure to chart it."
        />
        {progress.error && <ErrorBox error={progress.error} />}
        {progress.data ? (
          <ProgressPanel progress={progress.data} />
        ) : (
          !progress.error && <Loading />
        )}
      </section>
      <Card>
        <CardTitle>Workload (acute : chronic ratio)</CardTitle>
        {workload.error && <p className="text-sm text-bad">{workload.error}</p>}
        {workload.data ? (
          <WorkloadChart workload={workload.data} />
        ) : (
          !workload.error && <Loading />
        )}
        {workload.data?.latest?.flagged && (
          <p className="mt-2 text-sm text-bad">
            Workload ratio is high: consider lighter sessions.
          </p>
        )}
      </Card>
      <Card>
        <CardTitle>Sessions</CardTitle>
        {sessions.error && <p className="text-sm text-bad">{sessions.error}</p>}
        {sessions.data?.length === 0 && (
          <p className="text-sm text-muted">No sessions for {bowler.name} yet.</p>
        )}
        <ul className="divide-y divide-border">
          {(sessions.data ?? []).map((s) => (
            <li key={s.id} className="flex items-center justify-between py-2 text-sm">
              <Link href={`/sessions/${s.id}`} className="text-accent hover:underline">
                {s.original_name ?? s.id.slice(0, 8)}
              </Link>
              <span className="flex items-center gap-3 text-muted">
                {fmtDate(s.recorded_at ?? s.created_at)} <StatusBadge status={s.status} />
              </span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}

function NewBowler({ onCreated }: { onCreated: (b: BowlerRecord) => void }) {
  const [name, setName] = useState("");
  const [arm, setArm] = useState<"right" | "left">("right");
  const [type, setType] = useState("");
  const [height, setHeight] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      const b = await api.createBowler({
        name: name.trim(),
        bowling_arm: arm,
        bowling_type: type.trim() || null,
        height_cm: height ? Number(height) : null,
        dob: null,
      });
      setName("");
      setType("");
      setHeight("");
      onCreated(b);
    } catch (err) {
      setError(errorText(err));
    }
  }

  const input = "border border-border bg-surface-2 px-2 py-1.5";
  return (
    <Card>
      <CardTitle>New bowler</CardTitle>
      <form onSubmit={submit} className="grid grid-cols-2 gap-2 text-sm">
        <label className="col-span-2 flex flex-col gap-1">
          Name
          <input
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className={input}
          />
        </label>
        <label className="flex flex-col gap-1">
          Arm
          <select
            value={arm}
            onChange={(e) => setArm(e.target.value as "right" | "left")}
            className={input}
          >
            <option value="right">Right</option>
            <option value="left">Left</option>
          </select>
        </label>
        <label className="flex flex-col gap-1">
          Height (cm)
          <input
            type="number"
            min={100}
            max={230}
            value={height}
            onChange={(e) => setHeight(e.target.value)}
            className={input}
          />
        </label>
        <label className="col-span-2 flex flex-col gap-1">
          Type (e.g. fast, off-spin)
          <input value={type} onChange={(e) => setType(e.target.value)} className={input} />
        </label>
        <Button type="submit" className="col-span-2" disabled={!name.trim()}>
          Create profile
        </Button>
        {error && <p className="col-span-2 text-bad">{error}</p>}
      </form>
    </Card>
  );
}
