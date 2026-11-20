"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV: [string, string][] = [
  ["/", "Analyse"],
  ["/sessions", "Sessions"],
  ["/bowlers", "Bowlers"],
  ["/team", "Team"],
  ["/compare", "Compare actions"],
  ["/evaluation", "Model evaluation"],
  ["/settings", "Settings"],
];

function active(path: string, href: string): boolean {
  if (href === "/") return path === "/";
  if (href === "/sessions") return path.startsWith("/sessions") || path.startsWith("/deliveries");
  return path.startsWith(href);
}

/** Report-style masthead: wordmark, section navigation and the local engine's status. */
export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
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

  useReveal(path);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-foreground">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-end gap-x-10 gap-y-3 px-8 pb-3 pt-5">
          <Link href="/" className="group flex items-baseline gap-3">
            <span className="font-display text-[28px] leading-none">
              Bowl<span className="text-accent">S</span>ense
            </span>
            <span className="kicker hidden md:inline">Bowling biomechanics lab</span>
          </Link>
          <nav aria-label="Main" className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
            {NAV.map(([href, label]) => (
              <Link
                key={href}
                href={href}
                aria-current={active(path, href) ? "page" : undefined}
                className={cn(
                  "uline pb-1 transition-colors",
                  active(path, href) ? "text-foreground" : "text-muted hover:text-foreground",
                )}
              >
                {label}
              </Link>
            ))}
          </nav>
          <ThemeToggle />
          <p className="flex items-center gap-2 pb-1 text-xs text-muted" aria-live="polite">
            <span
              className={cn(
                "inline-block h-1.5 w-1.5 rounded-full",
                status === null ? "bg-muted" : status.api ? "bg-good" : "bg-bad",
              )}
            />
            {status === null
              ? "Connecting…"
              : status.api
                ? `Engine online · ${status.gpu ? "GPU" : "CPU"} · runs offline`
                : "Engine offline: start it with python run.py"}
          </p>
        </div>
      </header>
      {children}
    </div>
  );
}

/** Fade up every [data-reveal] element as it scrolls into view, including ones added later. */
function useReveal(path: string) {
  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.setAttribute("data-in", "");
            io.unobserve(e.target);
          }
        }),
      { rootMargin: "0px 0px -8% 0px", threshold: 0.05 },
    );
    const scan = () =>
      document.querySelectorAll("[data-reveal]:not([data-in])").forEach((el) => io.observe(el));
    scan();
    const mo = new MutationObserver(scan);
    mo.observe(document.body, { childList: true, subtree: true });
    return () => {
      io.disconnect();
      mo.disconnect();
    };
  }, [path]);
}

/** Light "paper" or dark "night nets" theme, remembered on this computer. */
function ThemeToggle() {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    setDark(document.documentElement.dataset.theme === "dark"); // eslint-disable-line react-hooks/set-state-in-effect
  }, []);
  function flip() {
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("bowlsense-theme", next);
    } catch {
      /* private window: the choice just isn't remembered */
    }
    setDark(!dark);
  }
  return (
    <button
      type="button"
      onClick={flip}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      className="uline ml-auto pb-1 text-xs text-muted hover:text-foreground"
    >
      {dark ? "Light" : "Dark"} mode
    </button>
  );
}
