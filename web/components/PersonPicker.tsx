"use client";

import { useMemo, useRef, useState } from "react";
import { api, errorText } from "@/lib/api";
import type { PeopleTracks } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { Button } from "./ui/Button";
import { ErrorBox, Loading } from "./ui/States";

/** Boxes of every tracked person at one frame (track id -> box). */
export function boxesAt(people: PeopleTracks, frame: number): Map<number, number[]> {
  const out = new Map<number, number[]>();
  for (const p of people.people) {
    const i = p.frames.indexOf(frame);
    if (i >= 0) out.set(p.track_id, p.boxes[i]);
  }
  return out;
}

/** The frame where the most people are visible (a good starting frame to pick from). */
export function busiestFrame(people: PeopleTracks): number {
  const counts = new Map<number, number>();
  for (const p of people.people) for (const f of p.frames) counts.set(f, (counts.get(f) ?? 0) + 1);
  let best = 0;
  let bestCount = -1;
  for (const [f, c] of counts) {
    if (c > bestCount || (c === bestCount && f < best)) {
      best = f;
      bestCount = c;
    }
  }
  return best;
}

/** "Wrong person?": show everyone the tracker followed and re-run with the one clicked. */
export function PersonPicker({ sessionId, onChosen }: { sessionId: string; onChosen: () => void }) {
  const people = useAsync(() => api.people(sessionId), [sessionId]);
  if (people.error) return <ErrorBox error={people.error} />;
  if (!people.data) return <Loading label="Loading tracked people…" />;
  return <Picker sessionId={sessionId} people={people.data} onChosen={onChosen} />;
}

function Picker({
  sessionId,
  people,
  onChosen,
}: {
  sessionId: string;
  people: PeopleTracks;
  onChosen: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [frame, setFrame] = useState(() => busiestFrame(people));
  const [selected, setSelected] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const boxes = useMemo(() => boxesAt(people, frame), [people, frame]);
  const last = Math.max(0, people.frame_count - 1);

  function seek(f: number) {
    const clamped = Math.min(last, Math.max(0, Math.round(f)));
    setFrame(clamped);
    if (video.current) video.current.currentTime = (clamped + 0.5) / people.fps;
  }

  async function confirm() {
    if (selected === null) return;
    setBusy(true);
    setError(null);
    try {
      await api.chooseBowler(sessionId, selected);
      onChosen();
    } catch (err) {
      setError(errorText(err));
      setBusy(false);
    }
  }

  const stroke = people.width / 300;
  return (
    <div className="space-y-3 text-sm">
      <p>
        Move the slider to a frame where the bowler is clearly visible, click their box, then
        re-run. The analysis then follows that person.
      </p>
      <div className="relative overflow-hidden border border-border bg-black">
        <video
          ref={video}
          src={api.videoUrl(sessionId)}
          preload="auto"
          muted
          playsInline
          className="block w-full"
          onLoadedMetadata={() => seek(frame)}
        />
        <svg
          viewBox={`0 0 ${people.width} ${people.height}`}
          className="absolute inset-0 h-full w-full"
          role="group"
          aria-label="Tracked people in this frame"
        >
          {[...boxes].map(([id, b]) => {
            const isSelected = id === selected;
            const isCurrent = id === people.chosen_track_id;
            const colour = isSelected ? "var(--good)" : isCurrent ? "var(--warn)" : "var(--accent)";
            return (
              <g
                key={id}
                role="button"
                tabIndex={0}
                aria-label={`Person ${id}${isCurrent ? " (currently analysed)" : ""}`}
                aria-pressed={isSelected}
                className="cursor-pointer"
                onClick={() => setSelected(id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === "") setSelected(id);
                }}
              >
                <rect
                  x={b[0]}
                  y={b[1]}
                  width={b[2] - b[0]}
                  height={b[3] - b[1]}
                  fill={isSelected ? "rgba(34,197,94,0.15)" : "rgba(0,0,0,0.01)"}
                  stroke={colour}
                  strokeWidth={isSelected ? stroke * 1.8 : stroke}
                />
                <text
                  x={b[0]}
                  y={Math.max(b[1] - stroke * 2, stroke * 8)}
                  fontSize={people.width / 45}
                  fill={colour}
                  fontWeight="700"
                >
                  {isCurrent ? `#${id} (analysed)` : `#${id}`}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex flex-1 items-center gap-2">
          <span className="whitespace-nowrap">Frame {frame}</span>
          <input
            type="range"
            min={0}
            max={last}
            value={frame}
            onChange={(e) => seek(Number(e.target.value))}
            className="flex-1"
            aria-label="Frame"
          />
        </label>
        <span className="text-muted">{boxes.size} people in this frame</span>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={confirm} disabled={selected === null || busy}>
          {busy
            ? "Re-running…"
            : selected === null
              ? "Click the bowler's box"
              : selected === people.chosen_track_id
                ? `Re-run with #${selected} (already analysed)`
                : `Re-run with #${selected}`}
        </Button>
        <span className="text-xs text-muted">
          Orange = analysed now (
          {people.chosen_by === "user" ? "your choice" : "picked automatically"})
        </span>
      </div>
      {error && <p className="text-bad">{error}</p>}
    </div>
  );
}
