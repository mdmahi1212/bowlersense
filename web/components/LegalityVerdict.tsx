import type { LegalityResult } from "@/lib/types";
import { cn } from "@/lib/utils";

type Verdict = LegalityResult["verdict"];

export const VERDICT_STYLE: Record<Verdict, { word: string; cls: string }> = {
  legal: { word: "LEGAL", cls: "border-good/50 bg-good/15 text-good" },
  not_legal: { word: "NOT LEGAL", cls: "border-bad/50 bg-bad/15 text-bad" },
  cannot_judge: { word: "CAN'T JUDGE", cls: "border-border bg-surface-2 text-muted" },
};

/** One verdict for a session: not legal if any delivery is, else legal if any could be judged. */
export function sessionVerdict(verdicts: Verdict[]): Verdict {
  if (verdicts.includes("not_legal")) return "not_legal";
  if (verdicts.includes("legal")) return "legal";
  return "cannot_judge";
}

/** The big verdict word, coloured and always written out in text. */
export function VerdictWord({ verdict, size = "lg" }: { verdict: Verdict; size?: "lg" | "sm" }) {
  const v = VERDICT_STYLE[verdict];
  return (
    <span
      data-testid="legality-verdict"
      className={cn(
        "inline-block rounded-md border font-extrabold tracking-wide",
        size === "lg" ? "px-3 py-1 text-2xl" : "px-2 py-0.5 text-xs",
        v.cls,
      )}
    >
      {v.word}
    </span>
  );
}
