"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const LINKS = [
  ["/", "Upload"],
  ["/sessions", "Sessions"],
  ["/bowlers", "Bowlers"],
  ["/compare", "Compare"],
  ["/settings", "Settings"],
] as const;

export function Nav() {
  const path = usePathname();
  return (
    <header className="border-b border-border bg-surface">
      <nav
        aria-label="Main"
        className="mx-auto flex max-w-6xl flex-wrap items-center gap-1 px-4 py-3"
      >
        <Link href="/" className="mr-4 text-lg font-bold tracking-tight">
          <span className="text-accent">Bowl</span>Sense
        </Link>
        {LINKS.map(([href, label]) => {
          const active = href === "/" ? path === "/" : path.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "rounded-md px-3 py-1.5 text-sm",
                active ? "bg-surface-2 text-foreground" : "text-muted hover:text-foreground",
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
