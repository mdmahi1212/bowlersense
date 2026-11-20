import type { SessionRecord } from "@/lib/types";
import { Badge } from "./ui/Badge";

const STYLE: Record<SessionRecord["status"], string> = {
  queued: "border-border bg-surface-2 text-muted",
  processing: "border-accent/50 bg-accent/15 text-accent",
  done: "border-good/40 bg-good/15 text-good",
  failed: "border-bad/40 bg-bad/15 text-bad",
};

export function StatusBadge({ status }: { status: SessionRecord["status"] }) {
  return <Badge className={STYLE[status]}>{status}</Badge>;
}
