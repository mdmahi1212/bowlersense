"use client";

import { useEffect, useRef, useState } from "react";
import { errorText } from "./api";

export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

interface Settled<T> {
  key: string | null;
  data: T | null;
  error: string | null;
}

/**
 * Run an async loader on mount and whenever `deps` change (or `reload()` is called). Errors
 * become text, never unhandled rejections; the previous data stays visible while reloading.
 */
export function useAsync<T>(load: () => Promise<T>, deps: readonly unknown[]): AsyncState<T> {
  const [tick, setTick] = useState(0);
  const key = JSON.stringify([...deps, tick]);
  const [settled, setSettled] = useState<Settled<T>>({ key: null, data: null, error: null });
  const loader = useRef(load);

  useEffect(() => {
    loader.current = load;
  });

  useEffect(() => {
    let live = true;
    loader.current().then(
      (data) => {
        if (live) setSettled({ key, data, error: null });
      },
      (err: unknown) => {
        if (live) setSettled((s) => ({ key, data: s.data, error: errorText(err) }));
      },
    );
    return () => {
      live = false;
    };
  }, [key]);

  return {
    data: settled.data,
    error: settled.key === key ? settled.error : null,
    loading: settled.key !== key,
    reload: () => setTick((t) => t + 1),
  };
}
