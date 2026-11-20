"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const LINKS = [
  ["/", "Upload"],
  ["/sessions", "Sessions"],
  ["/bowlers", "Bowlers"],
  ["/compare", "Compare"],
  ["/evaluation", "Evaluation"],
  ["/settings", "Settings"],
] as const;

export function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur">
      <nav
        aria-label="Main"
        className="mx-auto flex max-w-6xl flex-wrap items-center gap-1 px-4 py-3"
      >
        <Link href="/" className="mr-6 flex items-center gap-2" aria-label="BowlSense home">
          <span className="grid h-7 w-7 place-items-center rounded-sm bg-accent font-display text-sm font-bold text-black">
            BS
          </span>
          <span className="font-display text-lg font-bold uppercase tracking-[0.12em]">
            Bowl<span className="text-accent">Sense</span>
          </span>
          <span className="hidden rounded-sm border border-border px-1.5 py-0.5 font-display text-[10px] uppercase tracking-widest text-muted sm:inline">
            Analysis studio
          </span>
        </Link>
        {LINKS.map(([href, label]) => {
          const active = href === "/" ? path === "/" : path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "border-b-2 px-3 py-1.5 font-display text-sm uppercase tracking-wider",
                active
                  ? "border-accent text-foreground"
                  : "border-transparent text-muted hover:text-foreground",
              )}
            >
              {label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
