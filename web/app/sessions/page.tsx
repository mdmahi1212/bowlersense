"use client";

import Link from "next/link";
import { useState } from "react";
import { StatusBadge } from "@/components/StatusBadge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Empty, ErrorBox, Loading } from "@/components/ui/States";
import { api, errorText } from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { fmtDate } from "@/lib/utils";

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
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold">Sessions</h1>
        <label className="flex items-center gap-2 text-sm">
          Bowler
          <select
            value={bowlerId}
            onChange={(e) => setBowlerId(e.target.value)}
            className="rounded-md border border-border bg-surface-2 px-3 py-2"
          >
            <option value="">All</option>
            {(bowlers.data ?? []).map((b) => (
              <option key={b.id} value={b.id}>
                {b.name}
              </option>
            ))}
          </select>
        </label>
      </div>
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
        <Card className="overflow-x-auto p-0">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-border text-muted">
              <tr>
                <th className="p-3 font-medium">Video</th>
                <th className="p-3 font-medium">Bowler</th>
                <th className="p-3 font-medium">Uploaded</th>
                <th className="p-3 font-medium">Status</th>
                <th className="p-3" />
              </tr>
            </thead>
            <tbody>
              {sessions.data.map((s) => (
                <tr key={s.id} className="border-b border-border last:border-0">
                  <td className="p-3">
                    <Link
                      href={`/sessions/${s.id}`}
                      className="font-medium text-accent hover:underline"
                    >
                      {s.original_name ?? s.id.slice(0, 8)}
                    </Link>
                  </td>
                  <td className="p-3">{s.bowler_id ? (names.get(s.bowler_id) ?? "—") : "—"}</td>
                  <td className="p-3 text-muted">{fmtDate(s.created_at)}</td>
                  <td className="p-3">
                    <StatusBadge status={s.status} />
                  </td>
                  <td className="p-3 text-right">
                    <Button
                      variant="ghost"
                      onClick={() => remove(s.id)}
                      aria-label="Delete session"
                    >
                      Delete
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}
