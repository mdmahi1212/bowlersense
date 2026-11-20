"use client";

import { useRef, useState, type MouseEvent } from "react";
import { api, errorText } from "@/lib/api";
import { Button } from "./ui/Button";

const ORDER = [
  "Bowler's end, left end of the popping crease",
  "Bowler's end, right end of the popping crease",
  "Batter's end, right end of the popping crease",
  "Batter's end, left end of the popping crease",
];

/** Click the four crease corners on the first frame; saving re-runs the analysis. */
export function CalibrationPanel({
  sessionId,
  initial,
  onSaved,
}: {
  sessionId: string;
  initial: number[][] | null;
  onSaved: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [points, setPoints] = useState<number[][]>(initial ?? []);
  const [size, setSize] = useState<[number, number] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  function click(e: MouseEvent<HTMLDivElement>) {
    const v = video.current;
    if (!v || !size || points.length >= 4) return;
    const rect = v.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * size[0];
    const y = ((e.clientY - rect.top) / rect.height) * size[1];
    setPoints((p) => [...p, [Math.round(x * 10) / 10, Math.round(y * 10) / 10]]);
  }

  async function save() {
    if (points.length !== 4) return;
    setSaving(true);
    setError(null);
    try {
      await api.saveCalibration(sessionId, {
        points: points as [number[], number[], number[], number[]],
        rerun: true,
      });
      onSaved();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-3 text-sm">
      <p>
        {points.length < 4 ? (
          <>
            Click point {points.length + 1} of 4: <strong>{ORDER[points.length]}</strong>
          </>
        ) : (
          "All four points set. Save to re-run speed, length and line."
        )}
      </p>
      <div
        className="relative cursor-crosshair overflow-hidden rounded-md border border-border"
        onClick={click}
      >
        <video
          ref={video}
          src={api.videoUrl(sessionId)}
          preload="auto"
          muted
          playsInline
          className="block w-full"
          onLoadedMetadata={(e) =>
            setSize([e.currentTarget.videoWidth, e.currentTarget.videoHeight])
          }
        />
        {size && (
          <svg
            viewBox={`0 0 ${size[0]} ${size[1]}`}
            className="pointer-events-none absolute inset-0 h-full w-full"
          >
            {points.length > 1 && (
              <polygon
                points={points.map((p) => p.join(",")).join(" ")}
                fill="rgba(56,189,248,0.15)"
                stroke="var(--accent)"
                strokeWidth={size[0] / 400}
              />
            )}
            {points.map((p, i) => (
              <g key={i}>
                <circle cx={p[0]} cy={p[1]} r={size[0] / 120} fill="var(--accent)" />
                <text x={p[0] + size[0] / 80} y={p[1]} fontSize={size[0] / 40} fill="#fff">
                  {i + 1}
                </text>
              </g>
            ))}
          </svg>
        )}
      </div>
      <div className="flex gap-2">
        <Button variant="secondary" onClick={() => setPoints([])} disabled={points.length === 0}>
          Reset points
        </Button>
        <Button onClick={save} disabled={points.length !== 4 || saving}>
          {saving ? "Saving…" : "Save and re-run"}
        </Button>
      </div>
      {error && <p className="text-bad">{error}</p>}
    </div>
  );
}
