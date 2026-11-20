/** Hand-drawn line figures used as plates: a bowler at release and the filming set-up. */

const J: Record<string, [number, number]> = {
  head: [112, 22],
  neck: [110, 33],
  ls: [103, 39],
  rs: [117, 37],
  re: [125, 19],
  rw: [131, 3],
  le: [89, 47],
  lw: [78, 60],
  hip: [107, 80],
  lk: [88, 106],
  la: [75, 136],
  rk: [125, 107],
  ra: [141, 132],
  rt: [150, 135],
};
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
  ["ra", "rt"],
];

/** A bowler at the release point, drawn as a skeleton, with the elbow angle marked. */
export function BowlerAtRelease({ className }: { className?: string }) {
  return (
    <svg viewBox="40 -12 140 160" className={className} aria-hidden="true">
      <line x1="40" y1="138" x2="180" y2="138" stroke="currentColor" strokeOpacity="0.25" />
      <path
        d="M131 3 C 150 10, 162 30, 160 52"
        fill="none"
        stroke="var(--release)"
        strokeDasharray="2 3"
        strokeWidth="1.2"
      />
      {BONES.map(([a, b]) => (
        <line
          key={a + b}
          x1={J[a][0]}
          y1={J[a][1]}
          x2={J[b][0]}
          y2={J[b][1]}
          stroke={a === "rs" || a === "re" ? "var(--release)" : "currentColor"}
          strokeWidth="2.2"
          strokeLinecap="round"
        />
      ))}
      <circle
        cx={J.head[0]}
        cy={J.head[1]}
        r="7"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
      />
      {Object.entries(J)
        .filter(([k]) => k !== "head")
        .map(([k, [x, y]]) => (
          <circle key={k} cx={x} cy={y} r="1.8" fill="currentColor" />
        ))}
      <circle cx={J.rw[0]} cy={J.rw[1]} r="4.5" fill="none" stroke="var(--release)" />
      <path d="M 120.5 26 A 8 8 0 0 0 128.6 13.5" fill="none" stroke="var(--release)" />
      <text x="134" y="24" fontSize="7" fill="var(--release)" fontFamily="var(--font-mono)">
        171°
      </text>
    </svg>
  );
}

/** Top-down sketch of the recommended camera position (side-on, ~10 m, tripod). */
export function CameraSetup({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 320 170"
      className={className}
      role="img"
      aria-label="Camera set-up: side-on to the bowling crease, about 10 metres away, on a tripod"
    >
      <rect
        x="20"
        y="20"
        width="280"
        height="44"
        fill="none"
        stroke="currentColor"
        strokeOpacity="0.5"
      />
      <line x1="58" y1="20" x2="58" y2="64" stroke="currentColor" strokeOpacity="0.5" />
      <line x1="262" y1="20" x2="262" y2="64" stroke="currentColor" strokeOpacity="0.5" />
      <text x="24" y="14" fontSize="9" fill="currentColor" opacity="0.6">
        pitch, 20.12 m
      </text>
      <circle cx="70" cy="42" r="5" fill="var(--accent)" />
      <text x="62" y="80" fontSize="9" fill="var(--accent)">
        bowler
      </text>
      <line x1="70" y1="48" x2="70" y2="132" stroke="currentColor" strokeDasharray="3 3" />
      <text x="76" y="100" fontSize="9" fill="currentColor" fontFamily="var(--font-mono)">
        ≈10 m
      </text>
      <rect x="58" y="134" width="24" height="15" rx="2" fill="currentColor" />
      <circle cx="70" cy="141.5" r="4" fill="var(--background)" />
      <path d="M44 128 L70 141 L96 128" fill="none" stroke="currentColor" strokeOpacity="0.35" />
      <text x="92" y="146" fontSize="9" fill="currentColor">
        camera, side-on, hip height
      </text>
    </svg>
  );
}
