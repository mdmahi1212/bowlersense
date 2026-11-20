"use client";

import Link from "next/link";
import { useState } from "react";
import { StatusBadge } from "@/components/StatusBadge";
import { Button } from "@/components/ui/Button";
import { PageHead } from "@/components/ui/PageHead";
import { Empty, ErrorBox, Loading } from "@/components/ui/States";
import { api, errorText } from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { cleanName, fmtDate } from "@/lib/utils";

export default function SessionsPage() {
  const [bowlerId, setBowlerId] = useState("");
  const bowlers = useAsync(() => api.bowlers(), []);
  const sessions = useAsync(() => api.sessions(bowlerId || undefined), [bowlerId]);
  const [error, setError] = useState<string | null>(null);
  const names = new Map((bowlers.data ?? []).map((b) => [b.id, b.name]));

  async function remove(id: string) {
    if (!window.confirm("Delete this session and its files? This can't be undone.")) return;
    try {
      await api.deleteSession(id);
      sessions.reload();
    } catch (err) {
      setError(errorText(err));
    }
  }

  return (
    <div>
      <PageHead
        kicker="Archive"
        title="Sessions"
        deck="Every clip you have analysed on this computer, newest first."
        aside={
          <label className="flex items-center gap-3 text-sm">
            <span className="kicker">Bowler</span>
            <select
              value={bowlerId}
              onChange={(e) => setBowlerId(e.target.value)}
              className="border-b border-foreground/60 bg-transparent py-1.5 outline-none"
            >
              <option value="">All bowlers</option>
              {(bowlers.data ?? []).map((b) => (
                <option key={b.id} value={b.id}>
                  {b.name}
                </option>
              ))}
            </select>
          </label>
        }
      />
      {error && <ErrorBox error={error} />}
      {sessions.loading && !sessions.data && <Loading />}
      {sessions.error && (
        <ErrorBox
          error={sessions.error}
          action={<Button onClick={sessions.reload}>Retry</Button>}
        />
      )}
      {sessions.data && sessions.data.length === 0 && (
        <Empty title="No sessions yet">
          <Link href="/" className="text-accent underline">
            Upload a video
          </Link>{" "}
          to get started.
        </Empty>
      )}
      {sessions.data && sessions.data.length > 0 && (
        <ol className="border-t border-foreground">
          {sessions.data.map((s, i) => (
            <li
              key={s.id}
              className="group grid grid-cols-[3rem_minmax(0,1fr)_auto] items-baseline gap-x-6 gap-y-1 border-b border-border py-5 md:grid-cols-[3rem_minmax(0,1fr)_12rem_9rem_5rem]"
            >
              <span className="tabular text-xs text-muted">
                {String(sessions.data!.length - i).padStart(3, "0")}
              </span>
              <Link
                href={`/sessions/${s.id}`}
                className="truncate font-display text-2xl leading-snug group-hover:text-accent"
                title={s.original_name ?? undefined}
              >
                {cleanName(s.original_name)}
              </Link>
              <span className="hidden text-sm text-muted md:block">
                {s.bowler_id ? (names.get(s.bowler_id) ?? "Unknown bowler") : "No bowler set"}
                <br />
                {fmtDate(s.created_at)}
              </span>
              <span className="hidden md:block">
                <StatusBadge status={s.status} />
              </span>
              <button
                type="button"
                onClick={() => remove(s.id)}
                aria-label="Delete session"
                className="justify-self-end text-sm text-muted opacity-0 transition-opacity hover:text-bad focus:opacity-100 group-hover:opacity-100"
              >
                Delete
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
