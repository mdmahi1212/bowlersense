import type { Drill, Fault } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Button } from "./ui/Button";

function severityLabel(s: number): { label: string; cls: string } {
  if (s >= 66) return { label: "Major", cls: "border-bad/40 bg-bad/15 text-bad" };
  if (s >= 33) return { label: "Moderate", cls: "border-warn/40 bg-warn/15 text-warn" };
  return { label: "Minor", cls: "border-border bg-surface-2 text-foreground" };
}

/** Technique faults with severity, numbers, a jump-to-frame button and drills. */
export function FaultList({
  faults,
  drills,
  notMeasured,
  onJump,
}: {
  faults: Fault[];
  drills: Record<string, Drill[]> | null;
  notMeasured: Record<string, string>;
  onJump: (frame: number) => void;
}) {
  return (
    <div className="space-y-3">
      {faults.length === 0 && (
        <p className="text-sm text-good">
          No technique faults found in the checks this view allows.
        </p>
      )}
      {faults.map((f) => {
        const sev = severityLabel(f.severity);
        return (
          <div key={f.code} className="rounded-lg border border-border p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="font-semibold">{f.name}</p>
              <div className="flex items-center gap-2">
                <Badge className={sev.cls}>
                  {sev.label} · {f.severity.toFixed(0)}
                </Badge>
                {f.frames.length > 0 && (
                  <Button variant="secondary" onClick={() => onJump(f.frames[0])}>
                    Jump to frame {f.frames[0]}
                  </Button>
                )}
              </div>
            </div>
            <p className="mt-1 text-sm">{f.explanation}</p>
            {drills?.[f.code] && (
              <ul className="mt-2 space-y-1 text-sm">
                {drills[f.code].map((d) => (
                  <li key={d.name}>
                    <span className="font-medium">Drill: {d.name}</span>{" "}
                    <span className="text-muted">({d.reps})</span> — {d.description}
                  </li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
      {Object.keys(notMeasured).length > 0 && (
        <details className="text-sm text-muted">
          <summary className="cursor-pointer">
            Checks not possible from this view ({Object.keys(notMeasured).length})
          </summary>
          <ul className="mt-1 list-disc pl-5">
            {Object.entries(notMeasured).map(([code, why]) => (
              <li key={code}>
                {code}: {why}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
