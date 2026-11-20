import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("panel-hairline rounded-md border border-border bg-surface/90 p-4", className)}
      {...props}
    />
  );
}

export function CardTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-3 flex items-center justify-between gap-2">
      <h2 className="font-display text-[13px] font-semibold uppercase tracking-[0.14em] text-muted">
        <span className="mr-2 inline-block h-2 w-2 bg-accent align-middle" />
        {children}
      </h2>
      {aside}
    </div>
  );
}
