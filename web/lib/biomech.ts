/** Client-side biomechanics computed from a saved, cleaned PoseSequence (video-pixel
 *  coordinates, frame_idx = video frame). Distances are reported as % of body height so they
 *  don't depend on how far the camera was. */
import { type Arm, jointName } from "./pose";
import type { FramePose, Keypoint } from "./types";

export type XY = { x: number; y: number };
type Frames = Map<number, FramePose>;

const kp = (f: FramePose | undefined, name: string): Keypoint | null =>
  (f?.keypoints[name] as Keypoint | null | undefined) ?? null;

export function mid(a: XY | null, b: XY | null): XY | null {
  return a && b ? { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } : null;
}

export const hipMid = (f: FramePose | undefined) => mid(kp(f, "l_hip"), kp(f, "r_hip"));

/** Standing height in pixels: median nose-to-ankle distance over the frames x 1.07. */
export function bodyHeightPx(frames: Frames): number | null {
  const hs: number[] = [];
  for (const f of frames.values()) {
    const n = kp(f, "nose");
    const a = kp(f, "l_ankle") ?? kp(f, "r_ankle");
    if (n && a) hs.push(Math.abs(a.y - n.y));
  }
  if (hs.length === 0) return null;
  hs.sort((p, q) => p - q);
  const top = hs.slice(Math.floor(hs.length * 0.8)); // tallest 20%: upright running
  return (top[Math.floor(top.length / 2)] ?? hs[hs.length - 1]) * 1.07 || null;
}

/** Bowling-wrist position at release relative to the hip centre, in % of height
 *  (x positive = towards the batter's side the bowler faces, y positive = up). */
export function releasePoint(f: FramePose | undefined, arm: Arm, heightPx: number): XY | null {
  const w = kp(f, jointName(arm, "bowling", "wrist"));
  const h = hipMid(f);
  if (!w || !h || heightPx <= 0) return null;
  return { x: ((w.x - h.x) / heightPx) * 100, y: ((h.y - w.y) / heightPx) * 100 };
}

/** Mean and standard deviation of a list of points. */
export function spread(pts: XY[]): { mean: XY; sd: XY; radius: number } | null {
  if (pts.length === 0) return null;
  const m = {
    x: pts.reduce((a, p) => a + p.x, 0) / pts.length,
    y: pts.reduce((a, p) => a + p.y, 0) / pts.length,
  };
  const sd = {
    x: Math.sqrt(pts.reduce((a, p) => a + (p.x - m.x) ** 2, 0) / pts.length),
    y: Math.sqrt(pts.reduce((a, p) => a + (p.y - m.y) ** 2, 0) / pts.length),
  };
  return { mean: m, sd, radius: Math.hypot(sd.x, sd.y) };
}

/* Segment mass fractions (Dempster, simplified): [from, to, mass, centre along the segment]. */
const SEGMENTS: [string, string, number, number][] = [
  ["nose", "neck", 0.081, 0.5],
  ["neck", "hip", 0.497, 0.5],
  ["l_shoulder", "l_elbow", 0.028, 0.436],
  ["r_shoulder", "r_elbow", 0.028, 0.436],
  ["l_elbow", "l_wrist", 0.022, 0.5],
  ["r_elbow", "r_wrist", 0.022, 0.5],
  ["l_hip", "l_knee", 0.1, 0.433],
  ["r_hip", "r_knee", 0.1, 0.433],
  ["l_knee", "l_ankle", 0.061, 0.5],
  ["r_knee", "r_ankle", 0.061, 0.5],
];

function point(f: FramePose | undefined, name: string): XY | null {
  if (name === "neck") return mid(kp(f, "l_shoulder"), kp(f, "r_shoulder"));
  if (name === "hip") return hipMid(f);
  return kp(f, name);
}

/** Whole-body centre of mass (segment model); null when too few segments are visible. */
export function centreOfMass(f: FramePose | undefined): XY | null {
  let m = 0;
  let x = 0;
  let y = 0;
  for (const [a, b, mass, c] of SEGMENTS) {
    const p = point(f, a);
    const q = point(f, b);
    if (!p || !q) continue;
    m += mass;
    x += mass * (p.x + (q.x - p.x) * c);
    y += mass * (p.y + (q.y - p.y) * c);
  }
  return m >= 0.7 ? { x: x / m, y: y / m } : null;
}

/** Speed of every joint at ``frame`` (body heights per second, central difference). */
export function jointSpeeds(
  frames: Frames,
  frame: number,
  fps: number,
  heightPx: number,
): Record<string, number> {
  const a = frames.get(frame - 1);
  const b = frames.get(frame + 1);
  const out: Record<string, number> = {};
  if (!a || !b || heightPx <= 0) return out;
  for (const name of Object.keys(b.keypoints)) {
    const p = kp(a, name);
    const q = kp(b, name);
    if (p && q) out[name] = (Math.hypot(q.x - p.x, q.y - p.y) / heightPx) * (fps / 2);
  }
  return out;
}

/** Horizontal speed of the centre of mass (heights/s) per frame over [start, stop). */
export function comSpeeds(
  frames: Frames,
  start: number,
  stop: number,
  fps: number,
  heightPx: number,
) {
  const out: { frame: number; speed: number | null; com: XY | null }[] = [];
  for (let i = start; i < stop; i += 1) {
    const p = centreOfMass(frames.get(i - 1));
    const q = centreOfMass(frames.get(i + 1));
    out.push({
      frame: i,
      com: centreOfMass(frames.get(i)),
      speed: p && q && heightPx > 0 ? (Math.abs(q.x - p.x) / heightPx) * (fps / 2) : null,
    });
  }
  return out;
}

export interface FootContact {
  frame: number;
  foot: "l" | "r";
  x: number;
}

/** Foot contacts in the run-up: frames where an ankle reaches its lowest point (largest y)
 *  and is nearly still vertically, before ``until``. */
export function footContacts(
  frames: Frames,
  start: number,
  until: number,
  fps: number,
): FootContact[] {
  const out: FootContact[] = [];
  const gap = Math.max(2, Math.round(fps * 0.12));
  for (const foot of ["l", "r"] as const) {
    const name = `${foot}_ankle`;
    let last = -Infinity;
    for (let i = start + 2; i < until - 2; i += 1) {
      const ys = [-2, -1, 0, 1, 2].map((d) => kp(frames.get(i + d), name)?.y ?? null);
      if (ys.some((v) => v === null)) continue;
      const [a, b, c, d, e] = ys as number[];
      if (c >= a && c >= b && c >= d && c >= e && i - last >= gap) {
        out.push({ frame: i, foot, x: kp(frames.get(i), name)!.x });
        last = i;
      }
    }
  }
  return out.sort((p, q) => p.frame - q.frame);
}

/** Stride length (% height) and time between consecutive alternating contacts. */
export function runupStrides(contacts: FootContact[], fps: number, heightPx: number) {
  const strides: { frame: number; lengthPct: number; seconds: number }[] = [];
  for (let i = 1; i < contacts.length; i += 1) {
    const a = contacts[i - 1];
    const b = contacts[i];
    if (a.foot === b.foot || heightPx <= 0) continue;
    strides.push({
      frame: b.frame,
      lengthPct: (Math.abs(b.x - a.x) / heightPx) * 100,
      seconds: (b.frame - a.frame) / fps,
    });
  }
  return strides;
}

/** Head movement around release: the largest distance the nose moves from its release
 *  position within +-``windowS`` (% height), and a 0-100 stability score (100 = still). */
export function headStability(
  frames: Frames,
  release: number,
  fps: number,
  heightPx: number,
  windowS = 0.1,
): { movePct: number; score: number } | null {
  const r = kp(frames.get(release), "nose");
  if (!r || heightPx <= 0) return null;
  const n = Math.max(1, Math.round(windowS * fps));
  let worst = 0;
  for (let i = release - n; i <= release + n; i += 1) {
    const p = kp(frames.get(i), "nose");
    if (p) worst = Math.max(worst, Math.hypot(p.x - r.x, p.y - r.y));
  }
  const movePct = (worst / heightPx) * 100;
  return { movePct, score: Math.max(0, Math.min(100, 100 - movePct * 5)) };
}

/** The bowling wrist's path over [start, stop]. */
export function wristPath(frames: Frames, arm: Arm, start: number, stop: number): XY[] {
  const name = jointName(arm, "bowling", "wrist");
  const out: XY[] = [];
  for (let i = start; i <= stop; i += 1) {
    const w = kp(frames.get(i), name);
    if (w) out.push({ x: w.x, y: w.y });
  }
  return out;
}

/** 0-1 colour ramp ink -> amber -> red for joint speeds. */
export function speedColour(v: number, max: number): string {
  const t = Math.max(0, Math.min(1, v / Math.max(max, 1e-6)));
  const r = Math.round(60 + 195 * t);
  const g = Math.round(200 - 140 * t);
  const b = Math.round(255 - 215 * t);
  return `rgb(${r},${g},${b})`;
}
