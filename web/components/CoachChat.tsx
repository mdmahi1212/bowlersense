"use client";

import { useState, type FormEvent } from "react";
import { api, errorText } from "@/lib/api";
import type { ChatTurn } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { Button } from "./ui/Button";

/** Chat with the local coach about this session (falls back to rule-based answers offline). */
export function CoachChat({
  sessionId,
  context,
  placeholder,
}: {
  sessionId: string;
  /** Sent ahead of each question (not shown), e.g. the numbers of the paused frame. */
  context?: string;
  placeholder?: string;
}) {
  const history = useAsync(() => api.coachHistory(sessionId), [sessionId]);
  const [extra, setExtra] = useState<ChatTurn[]>([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const turns = [...(history.data ?? []), ...extra];

  async function send(e: FormEvent) {
    e.preventDefault();
    const text = message.trim();
    if (!text) return;
    setBusy(true);
    setError(null);
    setExtra((t) => [...t, { role: "user", content: text }]);
    setMessage("");
    try {
      const reply = await api.coach(
        sessionId,
        context
          ? `${context}

Question: ${text}`
          : text,
      );
      setExtra((t) => [...t, { role: "assistant", content: reply.reply }]);
      setNote(
        reply.source === "template"
          ? reply.note || "Local LLM unavailable: rule-based answer."
          : null,
      );
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="max-h-80 space-y-2 overflow-y-auto" aria-live="polite">
        {turns.length === 0 && (
          <p className="text-sm text-muted">
            Ask about your action, e.g. “How do I stop my front knee collapsing?”
          </p>
        )}
        {turns.map((t, i) => (
          <div
            key={i}
            className={
              t.role === "user"
                ? "ml-8 bg-accent/15 p-2 text-sm"
                : "mr-8 whitespace-pre-wrap bg-surface-2 p-2 text-sm"
            }
          >
            <span className="sr-only">{t.role === "user" ? "You:" : "Coach:"}</span>
            {t.content}
          </div>
        ))}
        {busy && (
          <p className="text-sm text-muted">Coach is thinking… (local models can take a minute)</p>
        )}
      </div>
      {note && <p className="text-xs text-muted">{note}</p>}
      {(error || history.error) && <p className="text-sm text-bad">{error ?? history.error}</p>}
      <form onSubmit={send} className="flex gap-2">
        <label htmlFor="coach-msg" className="sr-only">
          Message the coach
        </label>
        <input
          id="coach-msg"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          maxLength={2000}
          placeholder={placeholder ?? "Ask the coach…"}
          className="flex-1 border border-border bg-surface-2 px-3 py-2 text-sm"
        />
        <Button type="submit" disabled={busy || !message.trim()}>
          Send
        </Button>
      </form>
    </div>
  );
}
