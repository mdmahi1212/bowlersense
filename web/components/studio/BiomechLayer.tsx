"use client";

import { type XY, centreOfMass, jointSpeeds, speedColour, wristPath } from "@/lib/biomech";
import type { Arm } from "@/lib/pose";
import type { FramePose } from "@/lib/types";

export interface LayerFlags {
  speed: boolean;
  com: boolean;
  arm: boolean;
}

/** Analysis drawn over the video in video pixels: joint-speed heat map, centre-of-mass
 *  track, the bowling-wrist path (with a reference path ghosted) and highlighted joints. */
export function BiomechLayer({
  width,
  height,
  frames,
  frame,
  arm,
  fps,
  heightPx,
  window: [start, stop],
  release,
  show,
  focus,
  refPath,
}: {
  width: number;
  height: number;
  frames: Map<number, FramePose> | null;
  frame: number; // absolute video frame
  arm: Arm;
  fps: number;
  heightPx: number | null;
  window: [number, number];
  release: number;
  show: LayerFlags;
  focus: string[];
  refPath: XY[] | null;
}) {
  if (!frames) return null;
  const u = Math.max(width, height) / 400;
  const f = frames.get(frame);
  const path = show.arm ? wristPath(frames, arm, start, stop) : [];
  const relW = show.arm ? wristPath(frames, arm, release, release)[0] : undefined;
  const coms: XY[] = [];
  if (show.com)
    for (let i = start; i <= stop; i += 1) {
      const c = centreOfMass(frames.get(i));
      if (c) coms.push(c);
    }
  const com = show.com ? centreOfMass(f) : null;
  const speeds = show.speed && heightPx ? jointSpeeds(frames, frame, fps, heightPx) : {};
  const vmax = Math.max(1, ...Object.values(speeds));
  const fastest = Object.entries(speeds).sort((a, b) => b[1] - a[1])[0];
  const line = (pts: XY[]) => pts.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      className="pointer-events-none absolute inset-0 h-full w-full"
      aria-hidden="true"
    >
      {refPath && refPath.length > 1 && (
        <polyline
          points={line(refPath)}
          fill="none"
          stroke="rgba(120,200,255,0.75)"
          strokeWidth={1.6 * u}
          strokeDasharray={`${3 * u} ${3 * u}`}
        />
      )}
      {path.length > 1 && (
        <polyline
          points={line(path)}
          fill="none"
          stroke="#ff5b3a"
          strokeWidth={1.8 * u}
          strokeOpacity={0.9}
        />
      )}
      {relW && (
        <circle
          cx={relW.x}
          cy={relW.y}
          r={7 * u}
          fill="none"
          stroke="#ffc94a"
          strokeWidth={1.6 * u}
        />
      )}
      {coms.length > 1 && (
        <polyline
          points={line(coms)}
          fill="none"
          stroke="#ffd84d"
          strokeWidth={1.4 * u}
          strokeOpacity={0.7}
        />
      )}
      {com && (
        <g>
          <circle cx={com.x} cy={com.y} r={6 * u} fill="#ffd84d" stroke="#000" strokeWidth={u} />
          <line
            x1={com.x}
            y1={com.y}
            x2={com.x}
            y2={height}
            stroke="#ffd84d"
            strokeOpacity={0.4}
            strokeDasharray={`${2 * u} ${3 * u}`}
          />
          <text
            x={com.x + 9 * u}
            y={com.y - 6 * u}
            fontSize={9 * u}
            fill="#ffd84d"
            stroke="#000"
            strokeWidth={2.5 * u}
            paintOrder="stroke"
            fontFamily="var(--font-mono)"
          >
            COM
          </text>
        </g>
      )}
      {Object.entries(speeds).map(([name, v]) => {
        const k = f?.keypoints[name];
        if (!k) return null;
        return (
          <circle
            key={name}
            cx={k.x}
            cy={k.y}
            r={(3 + 9 * (v / vmax)) * u}
            fill={speedColour(v, vmax)}
            fillOpacity={0.85}
            stroke="#000"
            strokeWidth={0.6 * u}
          />
        );
      })}
      {fastest && f?.keypoints[fastest[0]] && (
        <text
          x={f.keypoints[fastest[0]]!.x + 12 * u}
          y={f.keypoints[fastest[0]]!.y + 4 * u}
          fontSize={10 * u}
          fill="#fff"
          stroke="#000"
          strokeWidth={2.5 * u}
          paintOrder="stroke"
          fontFamily="var(--font-mono)"
        >
          {fastest[1].toFixed(1)} h/s
        </text>
      )}
      {focus.map((name) => {
        const k = f?.keypoints[name];
        return k ? (
          <circle
            key={name}
            cx={k.x}
            cy={k.y}
            r={14 * u}
            fill="none"
            stroke="#ff3b30"
            strokeWidth={2.5 * u}
          >
            <animate
              attributeName="r"
              values={`${10 * u};${18 * u};${10 * u}`}
              dur="1.4s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="stroke-opacity"
              values="1;0.35;1"
              dur="1.4s"
              repeatCount="indefinite"
            />
          </circle>
        ) : null;
      })}
    </svg>
  );
}
