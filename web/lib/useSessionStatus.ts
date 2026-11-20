"use client";

import { useEffect, useRef, useState } from "react";
import { api, type SessionEvent } from "./api";

/**
 * Live session status: Server-Sent Events from the API, falling back to polling every 2 s if
 * the stream fails. Stops once the session is done or failed, calling `onFinish` once.
 */
export function useSessionStatus(
  id: string,
  initial: SessionEvent | null,
  onFinish?: () => void,
): SessionEvent | null {
  const [event, setEvent] = useState<SessionEvent | null>(initial);
  const finished = event?.status === "done" || event?.status === "failed";
  const finishRef = useRef(onFinish);
  useEffect(() => {
    finishRef.current = onFinish;
  });

  useEffect(() => {
    if (finished) return;
    let closed = false;
    let timer: ReturnType<typeof setInterval> | null = null;
    const update = (e: SessionEvent) => {
      if (closed) return;
      setEvent(e);
      if (e.status === "done" || e.status === "failed" || e.status === "deleted") {
        closed = true;
        finishRef.current?.();
      }
    };
    const poll = () => {
      api
        .session(id)
        .then((s) => {
          update({
            status: s.session.status,
            progress: s.job?.progress ?? 0,
            stage: s.job?.stage ?? null,
            attempts: s.job?.attempts ?? 0,
            error: s.session.error,
          });
        })
        .catch(() => {
          // keep polling; the page shows the last known state
        });
    };
    const startPolling = () => {
      if (!timer) timer = setInterval(poll, 2000);
    };
    let source: EventSource | null = null;
    if (typeof EventSource !== "undefined") {
      source = new EventSource(api.eventsUrl(id));
      source.onmessage = (msg) => {
        try {
          update(JSON.parse(msg.data) as SessionEvent);
        } catch {
          startPolling();
        }
      };
      source.onerror = () => {
        source?.close();
        startPolling();
      };
    } else {
      startPolling();
    }
    return () => {
      closed = true;
      source?.close();
      if (timer) clearInterval(timer);
    };
  }, [id, finished]);

  return event;
}
