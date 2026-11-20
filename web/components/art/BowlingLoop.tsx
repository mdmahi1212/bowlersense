"use client";

import { useEffect, useState } from "react";

/** A skeleton that bowls on a loop: bound, back-foot contact, front-foot contact, arm
 *  horizontal, release, follow-through. Joints are keyframed (hip-relative, x towards the
 *  batter is negative) and eased between poses; the bowling wrist leaves a fading trail. */

type P = [number, number];
type Pose = Record<string, P>;

const KEYS = ["head", "neck", "ls", "rs", "le", "re", "lw", "rw", "lk", "la", "rk", "ra"] as const;

const POSES: { name: string; x: number; y: number; t: number; p: Pose }[] = [
  {
    name: "Bound",
    x: 70,
    y: -10,
    t: 0.32,
    p: {
      head: [6, -72],
      neck: [4, -60],
      ls: [-3, -55],
      rs: [9, -55],
      le: [-14, -70],
      lw: [-20, -86],
      re: [14, -40],
      rw: [20, -26],
      lk: [-12, 20],
      la: [-4, 44],
      rk: [14, 18],
      ra: [6, 40],
    },
  },
  {
    name: "Back-foot contact",
    x: 38,
    y: 0,
    t: 0.26,
    p: {
      head: [13, -69],
      neck: [10, -57],
      ls: [2, -53],
      rs: [15, -54],
      le: [-12, -66],
      lw: [-27, -76],
      re: [23, -40],
      rw: [30, -26],
      lk: [-17, 26],
      la: [-32, 50],
      rk: [9, 27],
      ra: [11, 55],
    },
  },
  {
    name: "Front-foot contact",
    x: 8,
    y: 4,
    t: 0.16,
    p: {
      head: [7, -66],
      neck: [5, -54],
      ls: [-3, -50],
      rs: [11, -52],
      le: [-20, -48],
      lw: [-25, -33],
      re: [23, -38],
      rw: [33, -24],
      lk: [-22, 27],
      la: [-40, 54],
      rk: [18, 24],
      ra: [34, 46],
    },
  },
  {
    name: "Arm horizontal",
    x: 2,
    y: 5,
    t: 0.12,
    p: {
      head: [5, -63],
      neck: [4, -51],
      ls: [-4, -46],
      rs: [10, -48],
      le: [-20, -38],
      lw: [-28, -24],
      re: [28, -52],
      rw: [46, -55],
      lk: [-21, 27],
      la: [-40, 54],
      rk: [19, 25],
      ra: [34, 48],
    },
  },
  {
    name: "Release",
    x: -4,
    y: 6,
    t: 0.34,
    p: {
      head: [5, -58],
      neck: [3, -47],
      ls: [-4, -41],
      rs: [10, -43],
      le: [-18, -33],
      lw: [-29, -20],
      re: [18, -61],
      rw: [17, -80],
      lk: [-19, 26],
      la: [-40, 54],
      rk: [18, 27],
      ra: [34, 50],
    },
  },
  {
    name: "Follow-through",
    x: -34,
    y: 9,
    t: 0.7,
    p: {
      head: [-24, -46],
      neck: [-19, -37],
      ls: [-25, -32],
      rs: [-11, -37],
      le: [-8, -20],
      lw: [4, -8],
      re: [-27, -17],
      rw: [-36, 2],
      lk: [-21, 26],
      la: [-34, 55],
      rk: [-5, 24],
      ra: [2, 50],
    },
  },
];

const BONES: [string, string][] = [
  ["neck", "hip"],
  ["ls", "rs"],
  ["rs", "re"],
  ["re", "rw"],
  ["ls", "le"],
  ["le", "lw"],
  ["hip", "lk"],
  ["lk", "la"],
  ["hip", "rk"],
  ["rk", "ra"],
];

const CYCLE = POSES.reduce((a, p) => a + p.t, 0) + 0.5; // + pause before the next run
const ease = (u: number) => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);

function poseAt(sec: number): { j: Record<string, P>; phase: string; ms: number } {
  let t = sec % CYCLE;
  const relAt = POSES.slice(0, 4).reduce((a, p) => a + p.t, 0);
  const ms = ((sec % CYCLE) - relAt) * 1000;
  let i = 0;
  while (i < POSES.length - 1 && t > POSES[i].t) {
    t -= POSES[i].t;
    i += 1;
  }
  const a = POSES[i];
  const b = POSES[Math.min(i + 1, POSES.length - 1)];
  const u = i === POSES.length - 1 ? 0 : ease(Math.min(1, t / a.t));
  const mix = (p: P, q: P): P => [p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u];
  const [hx, hy] = mix([a.x, a.y], [b.x, b.y]);
  const j: Record<string, P> = { hip: [100 + hx, 92 + hy] };
  for (const k of KEYS) {
    const [x, y] = mix(a.p[k], b.p[k]);
    j[k] = [100 + hx + x, 92 + hy + y];
  }
  return { j, phase: u > 0.5 ? b.name : a.name, ms };
}

export function BowlingLoop({ className }: { className?: string }) {
  const [{ sec, trail }, setFrame] = useState<{ sec: number; trail: P[] }>({
    sec: POSES.slice(0, 4).reduce((a, p) => a + p.t, 0), // start at release
    trail: [],
  });
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const s = ((now - t0) / 1000) * 0.55; // 0.55x: a slow-motion feel
      const w = poseAt(s).j.rw;
      setFrame((f) => {
        const last = f.trail[f.trail.length - 1];
        const jumped = last && Math.hypot(w[0] - last[0], w[1] - last[1]) > 40; // loop restart
        return { sec: s, trail: jumped ? [w] : [...f.trail, w].slice(-26) };
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  const { j, phase, ms } = poseAt(sec);
  const atRelease = phase === "Release" && Math.abs(ms) < 90;

  return (
    <svg
      viewBox="20 -10 170 165"
      className={className}
      role="img"
      aria-label="Animated skeleton of a bowler running through the delivery"
    >
      <line x1="20" y1="148" x2="190" y2="148" stroke="currentColor" strokeOpacity="0.2" />
      {Array.from({ length: 18 }, (_, i) => (
        <line
          key={i}
          x1={20 + i * 10}
          y1="148"
          x2={20 + i * 10}
          y2="151"
          stroke="currentColor"
          strokeOpacity="0.2"
        />
      ))}
      {trail.length > 1 && (
        <polyline
          points={trail.map((p) => p.join(",")).join(" ")}
          fill="none"
          stroke="var(--release)"
          strokeWidth="1.4"
          strokeOpacity="0.7"
          strokeDasharray="2 2.5"
        />
      )}
      {BONES.map(([a, b]) => {
        const arm = a === "rs" || a === "re";
        return (
          <line
            key={a + b}
            x1={j[a][0]}
            y1={j[a][1]}
            x2={j[b][0]}
            y2={j[b][1]}
            stroke={arm ? "#ff5b3a" : "currentColor"}
            strokeWidth={arm ? 2.6 : 2.2}
            strokeLinecap="round"
          />
        );
      })}
      <circle
        cx={j.head[0]}
        cy={j.head[1] - 5}
        r="6.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      />
      {Object.entries(j).map(([k, [x, y]]) =>
        k === "head" ? null : <circle key={k} cx={x} cy={y} r="1.7" fill="currentColor" />,
      )}
      <circle
        cx={j.rw[0]}
        cy={j.rw[1]}
        r={atRelease ? 7 : 3.5}
        fill="none"
        stroke="#ffc94a"
        strokeOpacity={atRelease ? 1 : 0.6}
        style={{ transition: "r 120ms" }}
      />
      <text
        x="24"
        y="2"
        fontSize="6.5"
        fill="currentColor"
        opacity="0.55"
        fontFamily="var(--font-mono)"
        letterSpacing="0.5"
      >
        {phase.toUpperCase()}
      </text>
      <text
        x="186"
        y="2"
        fontSize="6.5"
        textAnchor="end"
        fill={atRelease ? "#ffc94a" : "currentColor"}
        opacity={atRelease ? 1 : 0.55}
        fontFamily="var(--font-mono)"
      >
        {ms >= 0 ? "+" : ""}
        {Math.round(ms)} ms
      </text>
    </svg>
  );
}
