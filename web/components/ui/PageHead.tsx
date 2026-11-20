import type { ReactNode } from "react";

/** Page masthead: a kicker, a large serif title and a one-line standfirst. */
export function PageHead({
  kicker,
  title,
  deck,
  aside,
}: {
  kicker?: ReactNode;
  title: ReactNode;
  deck?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-6 pb-8 pt-4">
      <div className="min-w-0 max-w-3xl">
        {kicker && <div className="kicker mb-3">{kicker}</div>}
        <h1 className="wipe font-display text-[52px] leading-[1.02] tracking-tight">{title}</h1>
        {deck && <p className="mt-3 text-[17px] leading-relaxed text-muted">{deck}</p>}
      </div>
      {aside && <div className="flex flex-wrap items-center gap-3 text-sm">{aside}</div>}
    </header>
  );
}
