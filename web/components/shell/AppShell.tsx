"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV: { section: string; items: [string, string, string][] }[] = [
  {
    section: "Analyse",
    items: [
      ["/", "New analysis", "M12 5v14M5 12h14"],
      ["/sessions", "Sessions", "M4 6h16M4 12h16M4 18h10"],
    ],
  },
  {
    section: "Players",
    items: [["/bowlers", "Bowlers", "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm-7 9a7 7 0 0 1 14 0"]],
  },
  {
    section: "Insight",
    items: [
      ["/compare", "Compare actions", "M8 4v16M16 4v16M4 8h4M16 16h4"],
      ["/evaluation", "Model evaluation", "M4 20V10M10 20V4M16 20v-7M22 20H2"],
    ],
  },
  {
    section: "System",
    items: [
      [
        "/settings",
        "Settings",
        "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm7.4-3a7.4 7.4 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7.6 7.6 0 0 0-2-1.2L14.5 2h-5l-.4 2.6a7.6 7.6 0 0 0-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 0 0 0 2.4l-2 1.6 2 3.4 2.4-1a7.6 7.6 0 0 0 2 1.2l.4 2.6h5l.4-2.6a7.6 7.6 0 0 0 2-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2Z",
      ],
    ],
  },
];

function Icon({ d }: { d: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-4 w-4 shrink-0"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={d} />
    </svg>
  );
}

/** Workstation layout: collapsible navigation rail + system status, content on the right. */
export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const [open, setOpen] = useState(true);
  const [status, setStatus] = useState<{ api: boolean; gpu: boolean } | null>(null);
  useEffect(() => {
    let live = true;
    const ping = () =>
      api.health().then(
        (h) => live && setStatus({ api: true, gpu: Boolean(h.gpu.torch_cuda && h.gpu.enabled) }),
        () => live && setStatus({ api: false, gpu: false }),
      );
    void ping();
    const t = setInterval(ping, 15000);
    return () => {
      live = false;
      clearInterval(t);
    };
  }, []);

  return (
    <div className="flex min-h-screen">
      <aside
        className={cn(
          "sticky top-0 flex h-screen shrink-0 flex-col border-r border-border bg-surface transition-[width]",
          open ? "w-56" : "w-14",
        )}
      >
        <div className="flex h-14 items-center gap-2 border-b border-border px-3">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-sm bg-accent font-display text-sm font-bold text-black">
            BS
          </span>
          {open && (
            <div className="leading-tight">
              <p className="font-display text-sm font-bold uppercase tracking-[0.14em]">
                BowlSense
              </p>
              <p className="text-[10px] uppercase tracking-widest text-muted">Biomechanics lab</p>
            </div>
          )}
        </div>
        <nav className="flex-1 overflow-y-auto py-3" aria-label="Main">
          {NAV.map((group) => (
            <div key={group.section} className="mb-3">
              {open && (
                <p className="px-4 pb-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-muted/70">
                  {group.section}
                </p>
              )}
              {group.items.map(([href, label, icon]) => {
                const active =
                  href === "/"
                    ? path === "/"
                    : path.startsWith(href) ||
                      (href === "/sessions" && path.startsWith("/deliveries"));
                return (
                  <Link
                    key={href}
                    href={href}
                    title={label}
                    aria-current={active ? "page" : undefined}
                    className={cn(
                      "mx-2 flex items-center gap-3 rounded-sm px-2.5 py-2 text-sm",
                      active
                        ? "bg-surface-2 text-foreground shadow-[inset_2px_0_0_var(--accent)]"
                        : "text-muted hover:bg-surface-2/60 hover:text-foreground",
                    )}
                  >
                    <Icon d={icon} />
                    {open && <span>{label}</span>}
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>
        <div className="border-t border-border p-3 text-[11px] text-muted">
          {open && (
            <ul className="mb-2 space-y-1">
              <li className="flex items-center gap-2">
                <span
                  className={cn("h-1.5 w-1.5 rounded-full", status?.api ? "bg-good" : "bg-bad")}
                />
                Engine {status === null ? "…" : status.api ? "online" : "offline"}
              </li>
              <li className="flex items-center gap-2">
                <span
                  className={cn("h-1.5 w-1.5 rounded-full", status?.gpu ? "bg-good" : "bg-warn")}
                />
                {status?.gpu ? "GPU accelerated" : "CPU mode"}
              </li>
              <li className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-good" />
                100% local
              </li>
            </ul>
          )}
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className="w-full rounded-sm border border-border py-1 hover:text-foreground"
            aria-label={open ? "Collapse navigation" : "Expand navigation"}
          >
            {open ? "«" : "»"}
          </button>
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">{children}</div>
    </div>
  );
}
