import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

/** A quiet raised sheet: hairline border, no shadow, square corners. */
export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("border border-border bg-surface p-5", className)} {...props} />;
}

export function CardTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-4 flex items-baseline justify-between gap-2">
      <h2 className="font-display text-xl">{children}</h2>
      {aside}
    </div>
  );
}
