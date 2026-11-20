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

/** The verdict word, coloured and always written out in text. ``verified={false}`` adds an
 *  "unverified" tag: the verdict follows the angle but the clip can't fully confirm it. */
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
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        data-testid="legality-verdict"
        className={cn(
          "inline-block rounded border font-bold tracking-wide",
          size === "lg" ? "px-2.5 py-0.5 text-lg" : "px-1.5 py-px text-[11px]",
          v.cls,
        )}
      >
        {v.word}
      </span>
      {!verified && (
        <span
          className={cn("text-warn", size === "lg" ? "text-sm" : "text-[11px]")}
          title="The verdict follows the measured angle, but this clip can't fully confirm it"
        >
          unverified
        </span>
      )}
    </span>
  );
}
