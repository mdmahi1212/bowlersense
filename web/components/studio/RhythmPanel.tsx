"use client";

import { useMemo } from "react";
import { comSpeeds, footContacts, runupStrides } from "@/lib/biomech";
import type { FramePose, PhaseEvents } from "@/lib/types";

const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

/** Run-up rhythm (stride lengths and timing from foot contacts) and how sharply the centre of
 *  mass brakes at front-foot contact. */
export function RhythmPanel({
  frames,
  clipStart,
  phases,
  fps,
  heightPx,
  onSeek,
}: {
  frames: Map<number, FramePose>;
  clipStart: number;
  phases: PhaseEvents;
  fps: number;
  heightPx: number;
  onSeek: (clipFrame: number) => void;
}) {
  const strides = useMemo(
    () =>
      runupStrides(
        footContacts(frames, clipStart + phases.run_up_start, clipStart + phases.bfc, fps),
        fps,
        heightPx,
      ),
    [frames, clipStart, phases, fps, heightPx],
  );
  const com = useMemo(
    () =>
      comSpeeds(
        frames,
        clipStart + phases.run_up_start,
        clipStart + phases.follow_through_end,
        fps,
        heightPx,
      ),
    [frames, clipStart, phases, fps, heightPx],
  );
  const ffc = clipStart + phases.ffc;
  const w = Math.max(2, Math.round(0.12 * fps));
  const before = avg(
    com
      .filter((c) => c.speed !== null && c.frame >= ffc - w && c.frame < ffc - 1)
      .map((c) => c.speed!),
  );
  const after = avg(
    com
      .filter((c) => c.speed !== null && c.frame > ffc + 1 && c.frame <= ffc + w)
      .map((c) => c.speed!),
  );
  const braking = before && after !== null ? ((before - after) / before) * 100 : null;
  const times = strides.map((s) => s.seconds);
  const tMean = avg(times);
  const cv =
    tMean && times.length > 1
      ? (Math.sqrt(avg(times.map((t) => (t - tMean) ** 2))!) / tMean) * 100
      : null;
  const lens = strides.map((s) => s.lengthPct);
  const trend = lens.length >= 4 ? avg(lens.slice(-2))! - avg(lens.slice(0, 2))! : null;
  const maxLen = Math.max(1, ...lens);
  const speeds = com.map((c) => c.speed ?? 0);
  const vmax = Math.max(0.01, ...speeds);
  const W = 600;
  const H = 120;

  return (
    <div className="grid gap-10 lg:grid-cols-2">
      <div>
        <p className="kicker">Run-up rhythm</p>
        {strides.length < 2 ? (
          <p className="mt-3 text-sm text-muted">
            Not enough of the run-up is in the clip to measure strides. Film from further back to
            include it.
          </p>
        ) : (
          <>
            <div
              className="mt-4 flex h-32 items-end gap-1.5"
              role="img"
              aria-label="Stride lengths through the run-up"
            >
              {strides.map((s, i) => (
                <button
                  key={s.frame}
                  type="button"
                  onClick={() => onSeek(s.frame - clipStart)}
                  title={`${s.lengthPct.toFixed(0)}% of height · ${(s.seconds * 1000).toFixed(0)} ms`}
                  className="grow-y flex-1 bg-foreground/80 transition-colors hover:bg-accent"
                  style={
                    {
                      height: `${(s.lengthPct / maxLen) * 100}%`,
                      "--d": `${i * 50}ms`,
                    } as React.CSSProperties
                  }
                />
              ))}
            </div>
            <dl className="mt-4 grid grid-cols-3 gap-4 text-sm">
              <div>
                <dt className="text-muted">Strides seen</dt>
                <dd className="font-display text-2xl">{strides.length}</dd>
              </div>
              <div>
                <dt className="text-muted">Timing variation</dt>
                <dd className="font-display text-2xl">{cv === null ? "—" : `${cv.toFixed(0)}%`}</dd>
              </div>
              <div>
                <dt className="text-muted">Stride trend</dt>
                <dd className="font-display text-2xl">
                  {trend === null
                    ? "—"
                    : trend > 3
                      ? "Lengthening"
                      : trend < -3
                        ? "Shortening"
                        : "Even"}
                </dd>
              </div>
            </dl>
            <p className="mt-2 text-xs text-muted">
              Bars are stride lengths (click to jump). A smooth run-up builds steadily with even
              timing; a stutter shows as a short bar.
            </p>
          </>
        )}
      </div>
      <div>
        <p className="kicker">Centre of mass: braking at front-foot contact</p>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="mt-4 h-32 w-full"
          preserveAspectRatio="none"
          role="img"
          aria-label="Centre-of-mass speed through the delivery"
        >
          <polyline
            points={com
              .map(
                (c, i) =>
                  `${(i / Math.max(1, com.length - 1)) * W},${H - ((c.speed ?? 0) / vmax) * (H - 8)}`,
              )
              .join(" ")}
            fill="none"
            stroke="var(--foreground)"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
          />
          {(() => {
            const i = com.findIndex((c) => c.frame === ffc);
            const x = (i / Math.max(1, com.length - 1)) * W;
            return i >= 0 ? (
              <line
                x1={x}
                x2={x}
                y1={0}
                y2={H}
                stroke="var(--accent)"
                strokeDasharray="4 4"
                vectorEffect="non-scaling-stroke"
              />
            ) : null;
          })()}
        </svg>
        <dl className="mt-4 grid grid-cols-3 gap-4 text-sm">
          <div>
            <dt className="text-muted">Before FFC</dt>
            <dd className="font-display text-2xl">
              {before === null ? "—" : `${before.toFixed(2)}`}
              <span className="ml-1 text-xs text-muted">h/s</span>
            </dd>
          </div>
          <div>
            <dt className="text-muted">After FFC</dt>
            <dd className="font-display text-2xl">
              {after === null ? "—" : `${after.toFixed(2)}`}
              <span className="ml-1 text-xs text-muted">h/s</span>
            </dd>
          </div>
          <div>
            <dt className="text-muted">Braking</dt>
            <dd className="font-display text-2xl text-accent">
              {braking === null ? "—" : `${braking.toFixed(0)}%`}
            </dd>
          </div>
        </dl>
        <p className="mt-2 text-xs text-muted">
          Speed of the body&apos;s centre of mass (body heights per second). A firm front leg stops
          the body sharply, passing the energy up to the arm.
        </p>
      </div>
    </div>
  );
}
