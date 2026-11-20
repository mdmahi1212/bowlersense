import type { ReactNode } from "react";

/** Section heading in the report style: an ink rule, an optional number, a serif title. */
export function SectionHead({
  title,
  hint,
  aside,
  n,
}: {
  title: string;
  hint?: string;
  aside?: ReactNode;
  n?: string;
}) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-x-6 gap-y-2 border-t border-foreground pt-3">
      <div>
        <h2 className="font-display text-2xl leading-tight">
          {n && <span className="tabular mr-3 align-middle text-xs text-accent">{n}</span>}
          {title}
        </h2>
        {hint && <p className="mt-0.5 text-sm text-muted">{hint}</p>}
      </div>
      {aside}
    </div>
  );
}
