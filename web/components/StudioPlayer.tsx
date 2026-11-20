"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { type Arm, angleAt, fmtDeg, jointName } from "@/lib/pose";
import type { FramePose, PhaseEvents } from "@/lib/types";
import { PHASE_EVENTS, cn } from "@/lib/utils";

export interface SeekRequest {
  frame: number; // clip frame
  nonce: number;
}

const SPEEDS = [0.1, 0.25, 0.5, 1] as const;
const LIME = "#c6f432";
const ORANGE = "#ff7a2f";

/** Frame-accurate review player: live skeleton, frame stepping, slow motion, phase timeline. */
export function StudioPlayer({
  videoSrc,
  overlaySrc,
  fps,
  realFps,
  clipStart,
  clipStop,
  phases,
  frames,
  arm,
  edges,
  seek,
  onFrame,
  onSetRelease,
  busy,
  width,
  height,
  hideTimeline = false,
}: {
  videoSrc: string;
  overlaySrc: string;
  fps: number;
  realFps: number;
  clipStart: number;
  clipStop: number;
  phases: PhaseEvents;
  frames: Map<number, FramePose> | null;
  arm: Arm;
  edges: string[][];
  seek: SeekRequest | null;
  onFrame: (clipFrame: number) => void;
  onSetRelease: (clipFrame: number) => void;
  busy: boolean;
  /** Video size in pixels (pose coordinates), used until the video's own size is known. */
  width: number;
  height: number;
  /** The page shows its own multi-track timeline instead. */
  hideTimeline?: boolean;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [frame, setFrame] = useState(phases.release);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(0.25);
  const [skeleton, setSkeleton] = useState(true);
  const [burned, setBurned] = useState(false);
  const [loop, setLoop] = useState(true);
  const [trail, setTrail] = useState(true);
  const [onion, setOnion] = useState(false);
  const length = Math.max(1, clipStop - clipStart);
  const last = length - 1;

  const draw = useCallback(
    (clipFrame: number) => {
      const c = canvas.current;
      const v = video.current;
      if (!c || !v) return;
      const w = v.videoWidth || width;
      const h = v.videoHeight || height;
      if (c.width !== w || c.height !== h) {
        c.width = w;
        c.height = h;
      }
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, w, h);
      const pose = frames?.get(clipStart + clipFrame);
      if (!skeleton || burned || !pose) return;
      const sc = Math.max(1.5, w / 500);
      const chain = ["shoulder", "elbow", "wrist"].map((p) => jointName(arm, "bowling", p));
      if (trail && frames) {
        // wrist path over the last 0.4 s (real time), fading
        const span = Math.max(4, Math.round(0.4 * realFps));
        let prev: { x: number; y: number } | null = null;
        for (let i = Math.max(0, clipFrame - span); i <= clipFrame; i++) {
          const wp = frames.get(clipStart + i)?.keypoints[chain[2]];
          if (wp && prev) {
            ctx.strokeStyle = `rgba(198,244,50,${(0.1 + 0.8 * (1 - (clipFrame - i) / span)).toFixed(2)})`;
            ctx.lineWidth = 2.2 * sc;
            ctx.beginPath();
            ctx.moveTo(prev.x, prev.y);
            ctx.lineTo(wp.x, wp.y);
            ctx.stroke();
          }
          prev = wp ?? null;
        }
      }
      if (onion && frames) {
        const step = Math.max(1, Math.round(0.02 * realFps));
        for (let g = 6; g >= 1; g--) {
          const kk = frames.get(clipStart + clipFrame - g * step)?.keypoints;
          if (!kk) continue;
          ctx.strokeStyle = `rgba(233,239,231,${(0.08 + 0.06 * (6 - g)).toFixed(2)})`;
          ctx.lineWidth = 2 * sc;
          for (const [a, b] of edges) {
            const p = kk[a];
            const q = kk[b];
            if (!p || !q) continue;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.stroke();
          }
        }
      }
      const k = pose.keypoints;
      const s = Math.max(1.5, w / 500);
      const bowl = new Set(["shoulder", "elbow", "wrist"].map((p) => jointName(arm, "bowling", p)));
      const front = new Set(
        ["hip", "knee", "ankle", "heel", "toe"].map((p) => jointName(arm, "front", p)),
      );
      ctx.lineCap = "round";
      for (const [a, b] of edges) {
        const p = k[a];
        const q = k[b];
        if (!p || !q) continue;
        ctx.strokeStyle =
          bowl.has(a) && bowl.has(b)
            ? LIME
            : front.has(a) && front.has(b)
              ? ORANGE
              : "rgba(233,239,231,0.85)";
        ctx.lineWidth =
          (bowl.has(a) && bowl.has(b)) || (front.has(a) && front.has(b)) ? 3.2 * s : 2 * s;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(q.x, q.y);
        ctx.stroke();
      }
      for (const kp of Object.values(k)) {
        if (!kp) continue;
        ctx.fillStyle = kp.conf >= 0.6 ? "#fff" : kp.conf >= 0.3 ? "#ffb020" : "#ff4d4d";
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 2.6 * s, 0, Math.PI * 2);
        ctx.fill();
      }
      const label = (at: string, text: string, colour: string) => {
        const p = k[at];
        if (!p) return;
        ctx.font = `600 ${Math.round(13 * s)}px Bahnschrift, Segoe UI, sans-serif`;
        const tw = ctx.measureText(text).width;
        const x = p.x + 10 * s;
        const y = p.y - 10 * s;
        ctx.fillStyle = "rgba(7,9,10,0.82)";
        ctx.fillRect(x - 4 * s, y - 14 * s, tw + 8 * s, 19 * s);
        ctx.fillStyle = colour;
        ctx.fillText(text, x, y);
      };
      const g = (role: "bowling" | "front", part: string) => k[jointName(arm, role, part)];
      label(
        jointName(arm, "bowling", "elbow"),
        `ELBOW ${fmtDeg(angleAt(g("bowling", "shoulder"), g("bowling", "elbow"), g("bowling", "wrist")))}`,
        LIME,
      );
      label(
        jointName(arm, "front", "knee"),
        `KNEE ${fmtDeg(angleAt(g("front", "hip"), g("front", "knee"), g("front", "ankle")))}`,
        ORANGE,
      );
    },
    [frames, clipStart, skeleton, burned, arm, edges, width, height, trail, onion, realFps],
  );

  const report = useCallback(
    (time: number) => {
      const f = Math.min(last, Math.max(0, Math.floor(time * fps + 1e-3) - clipStart));
      setFrame(f);
      onFrame(f);
      draw(f);
      const v = video.current;
      if (v && !v.paused && time * fps >= clipStop - 1) {
        if (loop) v.currentTime = (clipStart + 0.5) / fps;
        else v.pause();
      }
    },
    [fps, clipStart, clipStop, last, onFrame, draw, loop],
  );

  const goTo = useCallback(
    (f: number) => {
      const v = video.current;
      if (!v) return;
      v.pause();
      const clamped = Math.min(last, Math.max(0, Math.round(f)));
      v.currentTime = (clipStart + clamped + 0.5) / fps;
      report(v.currentTime);
    },
    [last, clipStart, fps, report],
  );

  // frame-accurate sync while playing
  useEffect(() => {
    const v = video.current as
      | (HTMLVideoElement & {
          requestVideoFrameCallback?: (cb: (n: number, m: { mediaTime: number }) => void) => number;
          cancelVideoFrameCallback?: (h: number) => void;
        })
      | null;
    if (!v || !playing) return;
    let handle = 0;
    let raf = 0;
    if (v.requestVideoFrameCallback) {
      const tick = (_: number, meta: { mediaTime: number }) => {
        report(meta.mediaTime);
        handle = v.requestVideoFrameCallback!(tick);
      };
      handle = v.requestVideoFrameCallback(tick);
      return () => v.cancelVideoFrameCallback?.(handle);
    }
    const loopRaf = () => {
      report(v.currentTime);
      raf = requestAnimationFrame(loopRaf);
    };
    raf = requestAnimationFrame(loopRaf);
    return () => cancelAnimationFrame(raf);
  }, [playing, report]);

  useEffect(() => {
    if (seek) goTo(seek.frame);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seek]);

  useEffect(() => {
    if (video.current) video.current.playbackRate = speed;
  }, [speed, burned]);

  useEffect(() => {
    draw(frame);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draw]);

  function toggle() {
    const v = video.current;
    if (!v) return;
    if (v.paused) {
      v.playbackRate = speed;
      void v.play().catch(() => setPlaying(false));
    } else v.pause();
  }

  function onKey(e: React.KeyboardEvent) {
    const step = e.shiftKey ? 10 : 1;
    const map: Record<string, () => void> = {
      " ": toggle,
      ArrowRight: () => goTo(frame + step),
      ArrowLeft: () => goTo(frame - step),
      "]": () =>
        setSpeed(
          SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(speed as (typeof SPEEDS)[number]) + 1)],
        ),
      "[": () =>
        setSpeed(SPEEDS[Math.max(0, SPEEDS.indexOf(speed as (typeof SPEEDS)[number]) - 1)]),
      r: () => goTo(phases.release),
    };
    const fn = map[e.key];
    if (fn) {
      e.preventDefault();
      fn();
    }
  }

  const pct = (f: number) => `${(Math.min(last, Math.max(0, f)) / last) * 100}%`;
  const t = (frame - phases.release) / realFps;
  const atRelease = frame === phases.release;

  return (
    <div
      className="space-y-3 outline-none"
      tabIndex={0}
      onKeyDown={onKey}
      aria-label="Delivery player. Space play/pause, arrows step frames, shift+arrows 10 frames, [ ] speed, R release"
    >
      <div
        className={cn(
          "relative overflow-hidden rounded-md border bg-black",
          atRelease ? "border-release" : "border-border",
        )}
      >
        <video
          ref={video}
          src={burned ? overlaySrc : videoSrc}
          playsInline
          muted
          preload="auto"
          className="block max-h-[62vh] w-full object-contain"
          onLoadedMetadata={() => goTo(frame)}
          onSeeked={(e) => report(e.currentTarget.currentTime)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onClick={toggle}
        />
        <canvas
          ref={canvas}
          className="pointer-events-none absolute inset-0 h-full w-full object-contain"
          style={{ objectFit: "contain" }}
        />
        <div className="pointer-events-none absolute left-3 top-3 flex gap-2">
          <span className="rounded-sm bg-black/70 px-2 py-1 font-display text-xs uppercase tracking-widest text-accent">
            {phaseAt(frame, phases)}
          </span>
          {atRelease && (
            <span className="rounded-sm bg-release px-2 py-1 font-display text-xs font-bold uppercase tracking-widest text-black">
              Release frame
            </span>
          )}
        </div>
        <div className="tabular pointer-events-none absolute bottom-3 right-3 rounded-sm bg-black/70 px-2 py-1 text-xs text-foreground">
          F {String(frame).padStart(4, "0")} / {last} · {t >= 0 ? "+" : ""}
          {(t * 1000).toFixed(0)} ms
        </div>
      </div>

      {!hideTimeline && (
        <Timeline length={length} frame={frame} phases={phases} onSeek={goTo} pct={pct} />
      )}

      <div className="flex flex-wrap items-center gap-1.5">
        <Ctl onClick={() => goTo(frame - 10)} label="Back 10 frames">
          ⏮ 10
        </Ctl>
        <Ctl onClick={() => goTo(frame - 1)} label="Previous frame">
          ◀ 1
        </Ctl>
        <Ctl onClick={toggle} label={playing ? "Pause" : "Play"} strong>
          {playing ? "❚❚ Pause" : "▶ Play"}
        </Ctl>
        <Ctl onClick={() => goTo(frame + 1)} label="Next frame">
          1 ▶
        </Ctl>
        <Ctl onClick={() => goTo(frame + 10)} label="Forward 10 frames">
          10 ⏭
        </Ctl>
        <span className="mx-1 h-6 w-px bg-border" />
        {SPEEDS.map((sp) => (
          <Ctl key={sp} onClick={() => setSpeed(sp)} active={speed === sp} label={`Speed ${sp}x`}>
            {sp}×
          </Ctl>
        ))}
        <span className="mx-1 h-6 w-px bg-border" />
        <Ctl onClick={() => goTo(phases.release)} label="Jump to release">
          ⦿ Release
        </Ctl>
        <button
          type="button"
          disabled={busy || atRelease}
          onClick={() => onSetRelease(frame)}
          className="rounded-sm border border-release px-2.5 py-1.5 font-display text-xs font-semibold uppercase tracking-wider text-release hover:bg-release hover:text-black disabled:opacity-40"
        >
          {busy ? "Re-analysing…" : "Set release here"}
        </button>
        <div className="ml-auto flex items-center gap-3 text-xs text-muted">
          <Toggle on={skeleton && !burned} set={setSkeleton} disabled={burned}>
            Skeleton
          </Toggle>
          <Toggle on={trail} set={setTrail} disabled={burned}>
            Wrist path
          </Toggle>
          <Toggle on={onion} set={setOnion} disabled={burned}>
            Onion skin
          </Toggle>
          <Toggle on={burned} set={setBurned}>
            Rendered overlay
          </Toggle>
          <Toggle on={loop} set={setLoop}>
            Loop
          </Toggle>
        </div>
      </div>
      <p className="text-xs text-muted">
        Keys: <kbd>Space</kbd> play · <kbd>←</kbd>/<kbd>→</kbd> frame · <kbd>Shift</kbd>+arrows 10
        frames · <kbd>[</kbd>/<kbd>]</kbd> speed · <kbd>R</kbd> release. Times are real time (
        {realFps.toFixed(0)} fps).
      </p>
    </div>
  );
}

export function phaseAt(f: number, p: PhaseEvents): string {
  if (f >= p.follow_through_end) return "Follow-through";
  if (f > p.release) return "Follow-through";
  if (f === p.release) return "Release";
  if (f >= p.arm_horizontal) return "Arm horizontal → release";
  if (f >= p.ffc) return "Front-foot contact";
  if (f >= p.bfc) return "Back-foot contact";
  if (f >= p.bound) return "Bound";
  return "Run-up";
}

function Timeline({
  length,
  frame,
  phases,
  onSeek,
  pct,
}: {
  length: number;
  frame: number;
  phases: PhaseEvents;
  onSeek: (f: number) => void;
  pct: (f: number) => string;
}) {
  const bar = useRef<HTMLDivElement>(null);
  const seekAt = (clientX: number) => {
    const r = bar.current?.getBoundingClientRect();
    if (r) onSeek(((clientX - r.left) / r.width) * (length - 1));
  };
  return (
    <div className="select-none">
      <div
        ref={bar}
        className="relative h-10 cursor-pointer rounded-sm border border-border bg-surface-2"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          seekAt(e.clientX);
        }}
        onPointerMove={(e) => e.buttons === 1 && seekAt(e.clientX)}
        role="slider"
        aria-label="Delivery timeline"
        aria-valuemin={0}
        aria-valuemax={length - 1}
        aria-valuenow={frame}
      >
        <div
          className="absolute inset-y-0 bg-accent/10"
          style={{
            left: pct(phases.bfc),
            width: `calc(${pct(phases.release)} - ${pct(phases.bfc)})`,
          }}
        />
        {PHASE_EVENTS.map(([key, label], i) => (
          <div key={key} className="absolute inset-y-0" style={{ left: pct(phases[key]) }}>
            <div
              className={cn("h-full w-px", key === "release" ? "w-0.5 bg-release" : "bg-muted/60")}
            />
            <span
              className={cn(
                "absolute ml-1 whitespace-nowrap font-display text-[10px] uppercase tracking-wider",
                i % 2 ? "bottom-0.5" : "top-0.5",
                key === "release" ? "text-release" : "text-muted",
              )}
            >
              {label}
            </span>
          </div>
        ))}
        <div
          className="absolute inset-y-0 w-0.5 bg-accent shadow-[0_0_8px_var(--accent)]"
          style={{ left: pct(frame) }}
        />
      </div>
    </div>
  );
}

function Ctl({
  children,
  onClick,
  label,
  active,
  strong,
}: {
  children: React.ReactNode;
  onClick: () => void;
  label: string;
  active?: boolean;
  strong?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cn(
        "tabular rounded-sm border px-2.5 py-1.5 text-xs transition-colors",
        strong
          ? "border-accent bg-accent font-semibold text-black hover:bg-accent-strong"
          : active
            ? "border-accent text-accent"
            : "border-border text-foreground hover:border-accent",
      )}
    >
      {children}
    </button>
  );
}

function Toggle({
  on,
  set,
  children,
  disabled,
}: {
  on: boolean;
  set: (v: boolean) => void;
  children: React.ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className={cn("flex cursor-pointer items-center gap-1.5", disabled && "opacity-40")}>
      <input
        type="checkbox"
        checked={on}
        disabled={disabled}
        onChange={(e) => set(e.target.checked)}
      />
      {children}
    </label>
  );
}
