import type { InjuryRisk } from "@/lib/types";
import { bandOf, bandStyle } from "@/lib/utils";

/** A front-view figure with each body region coloured (and labelled) by its risk component. */
export function BodyRiskMap({
  injury,
  bowlingArm,
}: {
  injury: InjuryRisk;
  bowlingArm: "right" | "left";
}) {
  const score = (name: string) => injury.components.find((c) => c.name === name)?.score ?? null;
  const colour = (name: string) => {
    const s = score(name);
    return s === null ? "var(--surface-2)" : bandStyle[bandOf(s)].fill;
  };
  // viewer's left is the bowler's right in a front view
  const armX = bowlingArm === "right" ? 38 : 122;
  const frontKneeX = bowlingArm === "right" ? 95 : 65;
  const regions = [
    ["lumbar", "Lower back"],
    ["knee", "Front knee"],
    ["shoulder", "Bowling shoulder"],
    ["side_strain", "Side (trunk)"],
  ] as const;
  return (
    <div className="flex flex-wrap items-center gap-4">
      <svg viewBox="0 0 160 240" className="h-60" role="img" aria-label="Body risk map">
        <circle cx="80" cy="22" r="14" fill="var(--surface-2)" stroke="var(--border)" />
        <rect
          x="56"
          y="40"
          width="48"
          height="58"
          rx="8"
          fill="var(--surface-2)"
          stroke="var(--border)"
        />
        <rect
          x="56"
          y="48"
          width="12"
          height="44"
          rx="5"
          fill={colour("side_strain")}
          opacity={0.85}
        />
        <rect
          x="92"
          y="48"
          width="12"
          height="44"
          rx="5"
          fill={colour("side_strain")}
          opacity={0.85}
        />
        <rect x="60" y="96" width="40" height="22" rx="6" fill={colour("lumbar")} />
        <circle cx={armX} cy="50" r="11" fill={colour("shoulder")} />
        <line
          x1={armX}
          y1="58"
          x2={armX}
          y2="110"
          stroke="var(--border)"
          strokeWidth="9"
          strokeLinecap="round"
        />
        <line
          x1={160 - armX}
          y1="58"
          x2={160 - armX}
          y2="110"
          stroke="var(--border)"
          strokeWidth="9"
          strokeLinecap="round"
        />
        <line
          x1="68"
          y1="118"
          x2="65"
          y2="225"
          stroke="var(--border)"
          strokeWidth="11"
          strokeLinecap="round"
        />
        <line
          x1="92"
          y1="118"
          x2="95"
          y2="225"
          stroke="var(--border)"
          strokeWidth="11"
          strokeLinecap="round"
        />
        <circle cx={frontKneeX} cy="172" r="10" fill={colour("knee")} />
      </svg>
      <ul className="space-y-1 text-sm">
        {regions.map(([name, label]) => {
          const s = score(name);
          const style = s === null ? null : bandStyle[bandOf(s)];
          return (
            <li key={name}>
              <span className="font-medium">{label}:</span>{" "}
              {style ? (
                <span className={style.text}>
                  {s!.toFixed(0)}/100 ({style.label})
                </span>
              ) : (
                <span className="text-muted">not measured</span>
              )}
            </li>
          );
        })}
        <li className="pt-1 text-xs text-muted">{injury.disclaimer}</li>
      </ul>
    </div>
  );
}
