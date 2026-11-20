import type { SessionRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

const STYLE: Record<SessionRecord["status"], [string, string]> = {
  queued: ["bg-muted", "Queued"],
  processing: ["bg-warn animate-pulse", "Analysing"],
  done: ["bg-good", "Analysed"],
  failed: ["bg-bad", "Failed"],
};

/** Status as a dot and a word (never colour alone). */
export function StatusBadge({ status }: { status: SessionRecord["status"] }) {
  const [dot, word] = STYLE[status] ?? ["bg-muted", status];
  return (
    <span className="inline-flex items-center gap-2 text-sm">
      <span className={cn("h-1.5 w-1.5 rounded-full", dot)} />
      {word}
    </span>
  );
}
