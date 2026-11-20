import type { LegalityResult } from "@/lib/types";
import { cn } from "@/lib/utils";

type Verdict = LegalityResult["verdict"];

export const VERDICT_STYLE: Record<Verdict, { word: string; cls: string }> = {
  legal: { word: "LEGAL", cls: "border-good/40 bg-good/10 text-good" },
  not_legal: { word: "NOT LEGAL", cls: "border-bad/40 bg-bad/10 text-bad" },
  cannot_judge: { word: "CAN'T JUDGE", cls: "border-border bg-surface-2 text-muted" },
};

/** One verdict for a session: not legal if any delivery is, else legal if any could be judged. */
export function sessionVerdict(verdicts: Verdict[]): Verdict {
  if (verdicts.includes("not_legal")) return "not_legal";
  if (verdicts.includes("legal")) return "legal";
  return "cannot_judge";
}

/** The verdict, always written out in words. ``lg`` is a serif headline; ``sm`` a dot + word.
 *  ``verified={false}`` adds "unverified": the verdict follows the angle, but the clip can't
 *  fully confirm it. */
export function VerdictWord({
  verdict,
  size = "lg",
  verified = true,
}: {
  verdict: Verdict;
  size?: "lg" | "sm";
  verified?: boolean;
}) {
  const v = VERDICT_STYLE[verdict];
  const colour =
    verdict === "legal" ? "text-good" : verdict === "not_legal" ? "text-bad" : "text-muted";
  if (size === "lg") {
    return (
      <span className="inline-flex items-baseline gap-3">
        <span
          data-testid="legality-verdict"
          className={cn(
            "font-display text-6xl leading-none lowercase first-letter:uppercase",
            colour,
          )}
        >
          {v.word}
        </span>
        {!verified && <span className="kicker !text-warn">Unverified</span>}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-sm">
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          verdict === "legal" ? "bg-good" : verdict === "not_legal" ? "bg-bad" : "bg-muted",
        )}
      />
      <span data-testid="legality-verdict" className={cn("font-medium", colour)}>
        {v.word}
      </span>
      {!verified && (
        <span
          className="text-xs text-warn"
          title="The verdict follows the measured angle, but this clip can't fully confirm it"
        >
          unverified
        </span>
      )}
    </span>
  );
}
