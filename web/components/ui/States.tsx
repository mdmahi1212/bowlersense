import type { ReactNode } from "react";
import { Card } from "./Card";

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-3 p-6 text-muted">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-accent border-t-transparent" />
      {label}
    </div>
  );
}

export function ErrorBox({ error, action }: { error: string; action?: ReactNode }) {
  return (
    <Card className="border-bad/50 bg-bad/10" role="alert">
      <p className="font-semibold text-bad">Something went wrong</p>
      <p className="mt-1 text-sm">{error}</p>
      {action && <div className="mt-3">{action}</div>}
    </Card>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <Card className="text-center">
      <p className="font-semibold">{title}</p>
      {children && <div className="mt-2 text-sm text-muted">{children}</div>}
    </Card>
  );
}
