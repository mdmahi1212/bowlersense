import { cn } from "@/lib/utils";

/** A 0-180 degree dial with a target band, the value written out in full. */
export function AngleDial({
  label,
  value,
  band,
  note,
  highlight,
}: {
  label: string;
  value: number | null;
  band?: [number, number];
  note?: string;
  highlight?: boolean;
}) {
  const pt = (deg: number, r: number) => {
    const a = Math.PI * (1 - Math.max(0, Math.min(180, deg)) / 180);
    return [60 + r * Math.cos(a), 60 - r * Math.sin(a)];
  };
  const arc = (from: number, to: number, r: number) => {
    const [x1, y1] = pt(from, r);
    const [x2, y2] = pt(to, r);
    return `M${x1},${y1} A${r},${r} 0 0 1 ${x2},${y2}`;
  };
  const inBand = value !== null && band ? value >= band[0] && value <= band[1] : null;
  const [nx, ny] = pt(value ?? 0, 44);
  return (
    <figure
      className={cn("border bg-surface-2 p-2", highlight ? "border-release" : "border-border")}
    >
      <svg
        viewBox="0 0 120 68"
        className="w-full"
        role="img"
        aria-label={`${label}: ${value === null ? "not measured" : `${value.toFixed(0)} degrees`}`}
      >
        <path d={arc(0, 180, 50)} fill="none" stroke="var(--border)" strokeWidth={6} />
        {band && (
          <path
            d={arc(band[0], band[1], 50)}
            fill="none"
            stroke="rgba(46,112,72,0.4)"
            strokeWidth={6}
          />
        )}
        {value !== null && (
          <>
            <line
              x1={60}
              y1={60}
              x2={nx}
              y2={ny}
              stroke={inBand === false ? "var(--warn)" : "var(--accent)"}
              strokeWidth={2.4}
              strokeLinecap="round"
            />
            <circle cx={60} cy={60} r={3} fill="var(--foreground)" />
          </>
        )}
        <text
          x={60}
          y={42}
          textAnchor="middle"
          fontSize={15}
          fontWeight={700}
          fill="var(--foreground)"
          fontFamily="var(--font-mono)"
        >
          {value === null ? "—" : `${value.toFixed(0)}°`}
        </text>
      </svg>
      <figcaption className="text-center">
        <p className="font-display text-[10px] uppercase tracking-widest text-muted">{label}</p>
        {note && (
          <p className={cn("text-[10px]", inBand === false ? "text-warn" : "text-muted")}>{note}</p>
        )}
      </figcaption>
    </figure>
  );
}
