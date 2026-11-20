/** Small shared helpers. */

/** Join class names, skipping falsy values (shadcn/ui's `cn`, without the merge step). */
export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(" ");
}

export type Band = "low" | "medium" | "high";

/** Text + colour classes for a risk band (colour is never the only signal). */
export const bandStyle: Record<Band, { label: string; text: string; bg: string; fill: string }> = {
  low: { label: "Low", text: "text-good", bg: "bg-good/15 border-good/40", fill: "var(--good)" },
  medium: {
    label: "Medium",
    text: "text-warn",
    bg: "bg-warn/15 border-warn/40",
    fill: "var(--warn)",
  },
  high: { label: "High", text: "text-bad", bg: "bg-bad/15 border-bad/40", fill: "var(--bad)" },
};

/** Band for a 0-100 risk score using the configured cut-offs. */
export function bandOf(score: number, low = 35, high = 65): Band {
  return score < low ? "low" : score > high ? "high" : "medium";
}

/** Fixed-precision number, or an em dash for missing values. */
export function fmt(value: number | null | undefined, digits = 0, unit = ""): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value.toFixed(digits)}${unit}`;
}

export function fmtBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

/** Phase event names in time order, with display labels. */
export const PHASE_EVENTS = [
  ["run_up_start", "Run-up"],
  ["bound", "Bound"],
  ["bfc", "BFC"],
  ["ffc", "FFC"],
  ["arm_horizontal", "Arm horizontal"],
  ["release", "Release"],
  ["follow_through_end", "Follow-through"],
] as const;

export type PhaseKey = (typeof PHASE_EVENTS)[number][0];

/** "jofra archer action slo-mo #cricket #shorts.mp4" -> "jofra archer action slo-mo" */
export function cleanName(name: string | null | undefined): string {
  const n = (name ?? "")
    .replace(/\.[a-z0-9]{2,4}$/i, "")
    .replace(/#\S+/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return n || "Untitled session";
}
