"use client";

import { useEffect, useState } from "react";

/** Counts a number up from zero when it first renders; non-numbers are shown as they are.
 *  With reduced motion the final value appears straight away. */
export function CountUp({
  value,
  decimals = 0,
  ms = 1100,
}: {
  value: string | number | null | undefined;
  decimals?: number;
  ms?: number;
}) {
  const target = typeof value === "number" ? value : Number.parseFloat(String(value ?? ""));
  const numeric = Number.isFinite(target);
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    if (!numeric) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const p = reduce ? 1 : Math.min(1, (now - t0) / ms);
      setProgress(p);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [numeric, target, ms]);
  if (!numeric) return <>{value ?? "—"}</>;
  return <>{(target * (1 - Math.pow(1 - progress, 3))).toFixed(decimals)}</>;
}
