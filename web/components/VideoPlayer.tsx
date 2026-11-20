"use client";

import { useEffect, useRef, useState } from "react";
import type { PhaseEvents } from "@/lib/types";
import { PHASE_EVENTS } from "@/lib/utils";
import { Button } from "./ui/Button";

export interface SeekRequest {
  frame: number; // clip frame
  nonce: number; // bump to seek to the same frame again
}

/**
 * Overlay/raw video of one delivery: frame stepping (← →), phase markers on the timeline and
 * the current clip frame reported to the page (for the synced angle charts).
 */
export function VideoPlayer({
  overlaySrc,
  rawSrc,
  fps,
  clipStart,
  clipStop,
  phases,
  seek,
  onFrame,
}: {
  overlaySrc: string;
  rawSrc: string;
  fps: number;
  clipStart: number;
  clipStop: number;
  phases: PhaseEvents;
  seek: SeekRequest | null;
  onFrame: (clipFrame: number) => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [overlay, setOverlay] = useState(true);
  const [frame, setFrame] = useState(phases.release);
  const resumeAt = useRef<number | null>(null);
  const length = Math.max(1, clipStop - clipStart);

  const goTo = (clipFrame: number) => {
    const v = video.current;
    if (!v) return;
    const f = Math.min(length - 1, Math.max(0, Math.round(clipFrame)));
    v.pause();
    v.currentTime = (clipStart + f + 0.5) / fps;
  };

  useEffect(() => {
    if (seek) goTo(seek.frame);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seek]);

  function report() {
    const v = video.current;
    if (!v) return;
    const f = Math.floor(v.currentTime * fps) - clipStart;
    if (v.currentTime * fps > clipStop && !v.paused) {
      v.pause();
      v.currentTime = (clipStart + 0.5) / fps; // loop the delivery
    }
    const clamped = Math.min(length - 1, Math.max(0, f));
    setFrame(clamped);
    onFrame(clamped);
  }

  function toggle() {
    resumeAt.current = video.current?.currentTime ?? null;
    setOverlay((o) => !o);
  }

  return (
    <div
      className="space-y-2"
      onKeyDown={(e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          goTo(frame + (e.key === "ArrowRight" ? 1 : -1));
        }
      }}
    >
      <video
        ref={video}
        src={overlay ? overlaySrc : rawSrc}
        controls
        playsInline
        preload="auto"
        className="w-full rounded-md bg-black"
        aria-label={overlay ? "Delivery with skeleton overlay" : "Original delivery video"}
        onLoadedMetadata={() => {
          const v = video.current;
          if (!v) return;
          v.currentTime = resumeAt.current ?? (clipStart + phases.release + 0.5) / fps;
          resumeAt.current = null;
        }}
        onTimeUpdate={report}
        onSeeked={report}
      />
      <div className="relative h-6 rounded bg-surface-2" aria-label="Phase timeline">
        {PHASE_EVENTS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            title={label}
            aria-label={`Jump to ${label}`}
            onClick={() => goTo(phases[key])}
            className="absolute top-0 h-6 w-1 -translate-x-1/2 bg-accent hover:bg-foreground"
            style={{ left: `${(phases[key] / length) * 100}%` }}
          />
        ))}
        <div
          className="pointer-events-none absolute top-0 h-6 w-0.5 bg-warn"
          style={{ left: `${(frame / length) * 100}%` }}
        />
      </div>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Button variant="secondary" onClick={() => goTo(frame - 1)} aria-label="Previous frame">
          ←
        </Button>
        <Button variant="secondary" onClick={() => goTo(frame + 1)} aria-label="Next frame">
          →
        </Button>
        <span className="tabular-nums text-muted">
          Frame {frame} / {length - 1} · {((frame - phases.release) / fps).toFixed(3)} s from
          release
        </span>
        <label className="ml-auto flex items-center gap-2">
          <input type="checkbox" checked={overlay} onChange={toggle} />
          Skeleton overlay
        </label>
      </div>
      <div className="flex flex-wrap gap-1 text-xs">
        {PHASE_EVENTS.map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => goTo(phases[key])}
            className="rounded border border-border px-2 py-0.5 hover:border-accent"
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
