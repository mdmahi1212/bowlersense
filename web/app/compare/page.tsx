"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { SkeletonView, boundsOf } from "@/components/SkeletonView";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { Empty, ErrorBox, Loading } from "@/components/ui/States";
import { api, errorText, fileUrl } from "@/lib/api";
import type { DeliveryOut, FramePose, PoseSequence, ReferencesOut } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";

export default function ComparePage() {
  return (
    <Suspense fallback={<Loading />}>
      <Compare />
    </Suspense>
  );
}

/** Frames from BFC to the end of the follow-through (the compared part of the action). */
function window(frames: FramePose[], from: number, to: number): FramePose[] {
  return frames.slice(Math.max(0, from), Math.min(frames.length, to + 1));
}

function Compare() {
  const query = useSearchParams();
  const refs = useAsync(() => api.references(), []);
  const sessions = useAsync(() => api.sessions(), []);
  const done = (sessions.data ?? []).filter((s) => s.status === "done");
  const [sessionId, setSessionId] = useState("");
  const [deliveryId, setDeliveryId] = useState(query.get("delivery") ?? "");
  const session = useAsync(
    () => (sessionId ? api.session(sessionId) : Promise.resolve(null)),
    [sessionId],
  );
  const delivery = useAsync(
    () => (deliveryId ? api.delivery(deliveryId) : Promise.resolve(null)),
    [deliveryId],
  );
  const [refName, setRefName] = useState("");

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Compare with a reference action</h1>
      <Card>
        <div className="flex flex-wrap items-end gap-3 text-sm">
          <label className="flex flex-col gap-1">
            Session
            <select
              value={sessionId}
              onChange={(e) => {
                setSessionId(e.target.value);
                setDeliveryId("");
              }}
              className="min-w-48 rounded-md border border-border bg-surface-2 px-3 py-2"
            >
              <option value="">{delivery.data ? "(from link)" : "Choose…"}</option>
              {done.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.original_name ?? s.id.slice(0, 8)}
                </option>
              ))}
            </select>
          </label>
          {session.data && (
            <label className="flex flex-col gap-1">
              Delivery
              <select
                value={deliveryId}
                onChange={(e) => setDeliveryId(e.target.value)}
                className="rounded-md border border-border bg-surface-2 px-3 py-2"
              >
                <option value="">Choose…</option>
                {session.data.delivery_ids.map((id, i) => (
                  <option key={id} value={id}>
                    Delivery {i + 1}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="flex flex-col gap-1">
            Reference
            <select
              value={refName}
              onChange={(e) => setRefName(e.target.value)}
              className="min-w-56 rounded-md border border-border bg-surface-2 px-3 py-2"
            >
              <option value="">Choose…</option>
              {(refs.data?.references ?? []).map((r) => (
                <option key={r.name} value={r.name}>
                  {r.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </Card>
      {(refs.error || sessions.error || delivery.error) && (
        <ErrorBox error={(refs.error ?? sessions.error ?? delivery.error)!} />
      )}
      {done.length === 0 && sessions.data && !delivery.data && (
        <Empty title="No finished sessions yet">Analyse a video first, then compare it here.</Empty>
      )}
      {delivery.data && refs.data && refName ? (
        <SideBySide out={delivery.data} refs={refs.data} refName={refName} />
      ) : (
        (delivery.data || refName) && (
          <p className="text-sm text-muted">Choose a delivery and a reference.</p>
        )
      )}
      {refs.data && (
        <Card>
          <CardTitle>About the references</CardTitle>
          <ul className="space-y-1 text-sm">
            {refs.data.references.map((r) => (
              <li key={r.name}>
                <span className="font-medium">{r.name}:</span> {r.description}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}

function SideBySide({
  out,
  refs,
  refName,
}: {
  out: DeliveryOut;
  refs: ReferencesOut;
  refName: string;
}) {
  const ref = refs.references.find((r) => r.name === refName)!;
  const d = out.delivery;
  const [pose, setPose] = useState<PoseSequence | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [mirror, setMirror] = useState(out.bowling_arm === "left");

  useEffect(() => {
    let live = true;
    fetch(fileUrl(out.pose))
      .then((r) =>
        r.ok
          ? (r.json() as Promise<PoseSequence>)
          : Promise.reject(new Error(`pose.json: ${r.status}`)),
      )
      .then((p) => live && setPose(p))
      .catch((e: unknown) => live && setError(errorText(e)));
    return () => {
      live = false;
    };
  }, [out.pose]);

  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setT((v) => (v >= 1 ? 0 : Math.min(1, v + 0.01))), 40);
    return () => clearInterval(timer);
  }, [playing]);

  const user = useMemo(() => {
    if (!pose) return [];
    const start = d.clip_start;
    const byIdx = new Map(pose.frames.map((f) => [f.frame_idx, f]));
    const clip: FramePose[] = [];
    for (let i = start + d.phases.bfc; i <= start + d.phases.follow_through_end; i++) {
      const f = byIdx.get(i);
      if (f) clip.push(f);
    }
    return clip;
  }, [pose, d]);
  const reference = useMemo(
    () => window(ref.sequence.frames, ref.phases.bfc, ref.phases.follow_through_end),
    [ref],
  );
  const userBounds = useMemo(() => boundsOf(user), [user]);
  const refBounds = useMemo(() => boundsOf(reference), [reference]);
  const pick = (frames: FramePose[]) =>
    frames.length ? frames[Math.round(t * (frames.length - 1))] : null;
  const comparison = d.comparisons.find((c) => c.name === refName);
  const maxDev = Math.max(1, ...Object.values(comparison?.deviation_deg ?? {}));

  if (error) return <ErrorBox error={error} />;
  if (!pose) return <Loading label="Loading keypoints…" />;
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardTitle>Your delivery {d.delivery_idx + 1}</CardTitle>
          <SkeletonView
            frame={pick(user)}
            bounds={userBounds}
            edges={refs.edges}
            colour="var(--accent)"
            mirror={mirror}
            label="Your skeleton"
          />
        </Card>
        <Card>
          <CardTitle>{ref.name}</CardTitle>
          <SkeletonView
            frame={pick(reference)}
            bounds={refBounds}
            edges={refs.edges}
            colour="var(--good)"
            label="Reference skeleton"
          />
        </Card>
      </div>
      <Card>
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <Button onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</Button>
          <label className="flex flex-1 items-center gap-2">
            <span className="sr-only">Position from back-foot contact to follow-through</span>
            <input
              type="range"
              min={0}
              max={1000}
              value={Math.round(t * 1000)}
              onChange={(e) => {
                setPlaying(false);
                setT(Number(e.target.value) / 1000);
              }}
              className="flex-1"
            />
          </label>
          <span className="tabular-nums text-muted">
            {Math.round(t * 100)}% BFC → follow-through
          </span>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={mirror} onChange={(e) => setMirror(e.target.checked)} />
            Mirror mine
          </label>
        </div>
        <p className="mt-2 text-xs text-muted">
          Both actions are time-normalised from back-foot contact to the end of the follow-through.
        </p>
      </Card>
      {comparison && (
        <Card>
          <CardTitle
            aside={
              <span className="text-sm">
                {comparison.similarity_pct.toFixed(0)}% similar overall
              </span>
            }
          >
            Per-joint deviation
          </CardTitle>
          <ul className="space-y-2 text-sm">
            {Object.entries(comparison.deviation_deg).map(([joint, dev]) => (
              <li key={joint} className="grid grid-cols-[9rem_1fr_4rem] items-center gap-2">
                <span>{joint.replace(/_/g, " ")}</span>
                <span className="h-3 rounded bg-surface-2">
                  <span
                    className="block h-3 rounded bg-accent"
                    style={{ width: `${(dev / maxDev) * 100}%` }}
                  />
                </span>
                <span className="text-right tabular-nums">{dev.toFixed(1)}°</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
