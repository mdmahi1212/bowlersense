import type { ReactNode } from "react";
import { Card } from "./Card";

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-3 p-6 text-muted">
      <span className="h-px w-10 animate-pulse bg-foreground" />
      {label}
    </div>
  );
}

export function ErrorBox({ error, action }: { error: string; action?: ReactNode }) {
  return (
    <Card className="border-l-2 border-l-bad" role="alert">
      <p className="font-display text-lg text-bad">Something went wrong</p>
      <p className="mt-1 text-sm">{error}</p>
      {action && <div className="mt-3">{action}</div>}
    </Card>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <Card className="py-10 text-center">
      <p className="font-display text-xl">{title}</p>
      {children && <div className="mt-2 text-sm text-muted">{children}</div>}
    </Card>
  );
}
