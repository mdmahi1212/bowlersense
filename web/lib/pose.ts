/** Client-side pose geometry: joint angles for any frame of a saved PoseSequence. */
import type { FramePose, Keypoint, PoseSequence } from "./types";

export type Arm = "right" | "left";
type Pt = Keypoint | null | undefined;

const side = (arm: Arm, role: "bowling" | "front" | "back") =>
  role === "bowling"
    ? arm === "right"
      ? "r"
      : "l"
    : role === "front"
      ? arm === "right"
        ? "l"
        : "r"
      : arm === "right"
        ? "r"
        : "l";

export const jointName = (arm: Arm, role: "bowling" | "front" | "back", part: string) =>
  `${side(arm, role)}_${part}`;

/** Angle at b (degrees, 180 = straight) between b->a and b->c. */
export function angleAt(a: Pt, b: Pt, c: Pt): number | null {
  if (!a || !b || !c) return null;
  const ux = a.x - b.x,
    uy = a.y - b.y,
    vx = c.x - b.x,
    vy = c.y - b.y;
  const n = Math.hypot(ux, uy) * Math.hypot(vx, vy);
  if (n === 0) return null;
  return (Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy) / n))) * 180) / Math.PI;
}

const mid = (a: Pt, b: Pt) =>
  a && b ? { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: null, conf: 1 } : null;

export interface FrameAngles {
  elbow: number | null;
  frontKnee: number | null;
  backKnee: number | null;
  trunkLean: number | null;
  upperArm: number | null;
  shoulderAbduction: number | null;
}

/** The key angles of one frame for a bowler with ``arm`` as the bowling arm. */
export function frameAngles(f: FramePose | undefined, arm: Arm): FrameAngles {
  const k = f?.keypoints ?? {};
  const g = (role: "bowling" | "front" | "back", part: string) => k[jointName(arm, role, part)];
  const sh = mid(k.l_shoulder, k.r_shoulder);
  const hp = mid(k.l_hip, k.r_hip);
  const bs = g("bowling", "shoulder"),
    be = g("bowling", "elbow");
  return {
    elbow: angleAt(bs, be, g("bowling", "wrist")),
    frontKnee: angleAt(g("front", "hip"), g("front", "knee"), g("front", "ankle")),
    backKnee: angleAt(g("back", "hip"), g("back", "knee"), g("back", "ankle")),
    trunkLean: sh && hp ? (Math.atan2(Math.abs(sh.x - hp.x), hp.y - sh.y) * 180) / Math.PI : null,
    upperArm: bs && be ? (Math.atan2(bs.y - be.y, Math.abs(be.x - bs.x)) * 180) / Math.PI : null,
    shoulderAbduction: angleAt(hp, bs, be),
  };
}

/** Frame lookup by video frame index. */
export function frameIndex(pose: PoseSequence): Map<number, FramePose> {
  return new Map(pose.frames.map((f) => [f.frame_idx, f]));
}

export const fmtDeg = (v: number | null | undefined, d = 0) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : `${v.toFixed(d)}°`;
