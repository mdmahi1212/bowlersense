"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, use, useMemo } from "react";
import { type Side, SyncCompare } from "@/components/studio/SyncCompare";
import { PageHead } from "@/components/ui/PageHead";
import { ErrorBox, Loading } from "@/components/ui/States";
import { api, fileUrl } from "@/lib/api";
import { frameIndex } from "@/lib/pose";
import type { PoseSequence } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { cleanName } from "@/lib/utils";

async function loadSide(id: string): Promise<Side> {
  const out = await api.delivery(id);
  const [pose, session] = await Promise.all([
    fetch(fileUrl(out.pose)).then((r) =>
      r.ok
        ? (r.json() as Promise<PoseSequence>)
        : Promise.reject(new Error(`skeleton ${r.status}`)),
    ),
    api.session(out.session_id),
  ]);
  return {
    out,
    frames: frameIndex(pose),
    label: `${cleanName(session.session.original_name)} · ball ${out.delivery.delivery_idx + 1}`,
  };
}

export default function VsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  return (
    <Suspense fallback={<Loading />}>
      <Vs id={id} />
    </Suspense>
  );
}

function Vs({ id }: { id: string }) {
  const other = useSearchParams().get("with");
  const router = useRouter();
  const a = useAsync(() => loadSide(id), [id]);
  const b = useAsync(() => (other ? loadSide(other) : Promise.resolve(null)), [other]);
  const refs = useAsync(() => api.references(), []);
  const sessions = useAsync(() => api.sessions(), []);
  const choices = useAsync(async () => {
    const done = (sessions.data ?? []).filter((s) => s.status === "done").slice(0, 25);
    const outs = await Promise.all(done.map((s) => api.session(s.id)));
    return outs.flatMap((o) =>
      o.delivery_ids.map((did, i) => ({
        id: did,
        label: `${cleanName(o.session.original_name)} · ball ${i + 1}`,
      })),
    );
  }, [sessions.data]);
  const options = useMemo(
    () => (choices.data ?? []).filter((c) => c.id !== id),
    [choices.data, id],
  );

  if (a.error) return <ErrorBox error={a.error} />;
  if (!a.data || !refs.data) return <Loading label="Loading the delivery…" />;
  return (
    <div className="space-y-8">
      <PageHead
        kicker={
          <>
            <Link href={`/deliveries/${id}`} className="hover:text-foreground">
              {a.data.label}
            </Link>{" "}
            / Side by side
          </>
        }
        title="Side by side, locked at release"
        deck="Both deliveries play in step from the release point, in real time. The other bowler's skeleton is ghosted over each video, scaled to the same body height."
        aside={
          <label className="flex items-center gap-3 text-sm">
            <span className="kicker">Compare with</span>
            <select
              value={other ?? ""}
              onChange={(e) => router.replace(`/deliveries/${id}/vs?with=${e.target.value}`)}
              className="max-w-72 border-b border-foreground/60 bg-transparent py-1.5 outline-none"
            >
              <option value="" disabled>
                {choices.loading ? "Loading deliveries…" : "Choose a delivery"}
              </option>
              {options.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
        }
      />
      {b.error && <ErrorBox error={b.error} />}
      {!other && (
        <p className="border-t border-foreground pt-6 font-display text-2xl text-muted">
          Choose a second delivery to line them up.
        </p>
      )}
      {other && !b.data && !b.error && <Loading label="Loading the second delivery…" />}
      {b.data && <SyncCompare a={a.data} b={b.data} edges={refs.data.edges} />}
    </div>
  );
}
