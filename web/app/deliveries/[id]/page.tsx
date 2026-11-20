"use client";

import Link from "next/link";
import { use, useCallback, useMemo, useState } from "react";
import { AngleChart } from "@/components/AngleChart";
import { BodyRiskMap } from "@/components/BodyRiskMap";
import { FaultList } from "@/components/FaultList";
import { VERDICT_STYLE, VerdictWord } from "@/components/LegalityVerdict";
import { RiskGauge } from "@/components/RiskGauge";
import { StudioPlayer, phaseAt, type SeekRequest } from "@/components/StudioPlayer";
import {
  KineticChain,
  LiveAngles,
  MechanicsPanel,
  PhaseFilmstrip,
} from "@/components/StudioPanels";
import { AngleDial } from "@/components/studio/AngleDial";
import { ReleaseLoop } from "@/components/studio/ReleaseLoop";
import { type Lane, TrackDock } from "@/components/studio/TrackDock";
import { ErrorBox, Loading } from "@/components/ui/States";
import { api, errorText, fileUrl } from "@/lib/api";
import { type Arm, frameAngles, frameIndex } from "@/lib/pose";
import type { DeliveryOut, PoseSequence } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { cn, fmt } from "@/lib/utils";

const TABS = ["Release", "Live", "Mechanics", "Legality", "Faults", "Load"] as const;
type Tab = (typeof TABS)[number];
const JOINTS = [
  ["bowling_elbow", "Elbow"],
  ["front_knee", "Front knee"],
  ["back_knee", "Back knee"],
  ["trunk_lean", "Trunk"],
  ["upper_arm_elevation", "Upper arm"],
] as const;

export default function DeliveryPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const delivery = useAsync(() => api.delivery(id), [id]);
  const refs = useAsync(() => api.references(), []);
  if (delivery.error) return <ErrorBox error={delivery.error} />;
  if (!delivery.data || !refs.data) return <Loading label="Opening the workspace…" />;
  return (
    <Workspace key={id} out={delivery.data} edges={refs.data.edges} reload={delivery.reload} />
  );
}

function Workspace({
  out,
  edges,
  reload,
}: {
  out: DeliveryOut;
  edges: string[][];
  reload: () => void;
}) {
  const d = out.delivery;
  const arm = out.bowling_arm as Arm;
  const pose = useAsync(
    () =>
      fetch(fileUrl(out.pose)).then((r) =>
        r.ok
          ? (r.json() as Promise<PoseSequence>)
          : Promise.reject(new Error(`skeleton ${r.status}`)),
      ),
    [out.pose],
  );
  const session = useAsync(() => api.session(out.session_id), [out.session_id]);
  const drills = useAsync(() => api.drills(), []);
  const frames = useMemo(() => (pose.data ? frameIndex(pose.data) : null), [pose.data]);
  const [frame, setFrame] = useState(d.phases.release);
  const [seek, setSeek] = useState<SeekRequest | null>(null);
  const [tab, setTab] = useState<Tab>("Release");
  const [joint, setJoint] = useState<(typeof JOINTS)[number][0]>("bowling_elbow");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const jump = useCallback(
    (f: number) => setSeek((s) => ({ frame: f, nonce: (s?.nonce ?? 0) + 1 })),
    [],
  );
  const onFrame = useCallback((f: number) => setFrame(f), []);
  const clipStop = d.clip_stop ?? d.clip_start + d.phases.follow_through_end + 1;
  const length = clipStop - d.clip_start;

  async function setRelease(f: number) {
    setBusy(true);
    setError(null);
    try {
      await api.setRelease(out.delivery_id, f);
      reload();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  const curve = (name: string) => d.biomech_curves.find((c) => c.name === name);
  const values = (name: string, abs = false) =>
    Array.from({ length }, (_, i) => {
      const v = curve(name)?.values[i];
      return v === null || v === undefined ? null : abs ? Math.abs(v) : v;
    });
  const lanes: Lane[] = [
    { label: "Elbow", unit: "°", values: values("bowling_elbow"), colour: "#c6f432" },
    { label: "Front knee", unit: "°", values: values("front_knee"), colour: "#ff7a2f" },
    { label: "Trunk lean", unit: "°", values: values("trunk_lean"), colour: "#7cc7ff" },
    {
      label: "Arm speed",
      unit: "°/s",
      values: values("upper_arm_elevation_velocity", true),
      colour: "#e9efe7",
    },
  ];
  const markers = d.faults.flatMap((f) =>
    f.frames.slice(0, 1).map((fr) => ({ frame: fr, label: f.name })),
  );
  const ids = session.data?.delivery_ids ?? [];
  const pos = ids.indexOf(out.delivery_id);
  const rel = frameAngles(frames?.get(d.clip_start + d.phases.release), arm);
  const bm = new Map(d.biomechanics.map((m) => [m.key, m.value]));
  const verdict = VERDICT_STYLE[d.legality.verdict];
  const src = api.videoUrl(out.session_id);
  const releaseMarkers = [
    { frame: d.phases.ffc, label: "FFC" },
    { frame: d.phases.release, label: "REL" },
  ];

  return (
    <div className="-mx-6 -my-5 flex min-h-[calc(100vh-2.5rem)] flex-col">
      <div className="sticky top-0 z-20 flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-border bg-background/95 px-5 py-2.5 backdrop-blur">
        <nav className="flex items-center gap-1.5 text-xs text-muted" aria-label="Breadcrumb">
          <Link href="/sessions" className="hover:text-foreground">
            Sessions
          </Link>
          <span>/</span>
          <Link
            href={`/sessions/${out.session_id}`}
            className="max-w-48 truncate hover:text-foreground"
          >
            {session.data?.session.original_name ?? "Session"}
          </Link>
          <span>/</span>
          <h1 className="font-display text-sm font-semibold uppercase tracking-wider text-foreground">
            Delivery {d.delivery_idx + 1} of {out.delivery_count}
          </h1>
        </nav>
        <div className="flex gap-1">
          <NavBtn href={pos > 0 ? `/deliveries/${ids[pos - 1]}` : null}>‹ Prev</NavBtn>
          <NavBtn href={pos >= 0 && pos < ids.length - 1 ? `/deliveries/${ids[pos + 1]}` : null}>
            Next ›
          </NavBtn>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <VerdictWord verdict={d.legality.verdict} size="sm" />
          <Chip label="Technique" value={`${d.technique_score.toFixed(0)}/100`} />
          <Chip label="Risk" value={`${d.injury.overall.toFixed(0)} ${d.injury.band}`} />
          <Chip label="Arm speed" value={`${fmt(bm.get("arm_angular_velocity"), 0)}°/s`} />
          <Chip
            label="Release"
            value={`${d.release_source === "user" ? "manual" : "auto"} · f${d.phases.release}`}
          />
          <a
            href={api.reportUrl(out.session_id)}
            download
            className="rounded-sm border border-border px-2.5 py-1 text-xs hover:border-accent"
          >
            PDF
          </a>
        </div>
      </div>
      {(error || pose.error) && (
        <div className="px-5 pt-3">
          <ErrorBox error={error ?? `Couldn't load the skeleton: ${pose.error}`} />
        </div>
      )}

      <div className="grid flex-1 xl:grid-cols-[minmax(0,1fr)_400px]">
        <section className="min-w-0 space-y-3 p-4" aria-label="Viewer">
          <StudioPlayer
            videoSrc={src}
            overlaySrc={fileUrl(out.overlay)}
            fps={out.fps}
            realFps={out.analysis_fps}
            clipStart={d.clip_start}
            clipStop={clipStop}
            phases={d.phases}
            frames={frames}
            arm={arm}
            edges={edges}
            seek={seek}
            onFrame={onFrame}
            onSetRelease={setRelease}
            busy={busy}
            width={out.width}
            height={out.height}
            hideTimeline
          />
          <TrackDock
            length={length}
            frame={frame}
            phases={d.phases}
            lanes={lanes}
            markers={markers}
            onSeek={jump}
            fps={out.analysis_fps}
          />
        </section>

        <aside className="border-l border-border bg-surface/60" aria-label="Inspector">
          <div
            className="sticky top-[49px] z-10 flex border-b border-border bg-surface"
            role="tablist"
          >
            {TABS.map((t) => (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={tab === t}
                onClick={() => setTab(t)}
                className={cn(
                  "flex-1 border-b-2 px-1 py-2.5 font-display text-[11px] uppercase tracking-wider",
                  tab === t
                    ? "border-accent text-foreground"
                    : "border-transparent text-muted hover:text-foreground",
                )}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="space-y-4 p-4">
            {tab === "Release" && (
              <>
                <ReleaseLoop
                  videoSrc={src}
                  fps={out.fps}
                  realFps={out.analysis_fps}
                  releaseFrame={d.clip_start + d.phases.release}
                  frames={frames}
                  arm={arm}
                  edges={edges}
                  width={out.width}
                  height={out.height}
                />
                <div className="grid grid-cols-2 gap-2">
                  <AngleDial
                    label="Elbow at release"
                    value={rel.elbow}
                    band={[160, 180]}
                    note="near straight"
                    highlight
                  />
                  <AngleDial
                    label="Front knee"
                    value={rel.frontKnee}
                    band={[160, 180]}
                    note="braced ≥ 160°"
                  />
                  <AngleDial
                    label="Trunk lean"
                    value={rel.trunkLean}
                    band={[0, 35]}
                    note="from vertical"
                  />
                  <AngleDial
                    label="Arm elevation"
                    value={rel.upperArm === null ? null : rel.upperArm + 90}
                    band={[150, 180]}
                    note="0 = down, 180 = up"
                  />
                </div>
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  <Kv
                    k="Elbow straightening"
                    v={
                      d.legality.screened
                        ? fmt(d.legality.elbow_extension_deg, 1, "°")
                        : "not screened"
                    }
                  />
                  <Kv k="Release height" v={fmt(d.release_height_pct, 0, "% height")} />
                  <Kv k="Arm horiz. → release" v={fmt(bm.get("ah_to_release"), 3, " s")} />
                  <Kv k="Wrist speed" v={fmt(bm.get("wrist_speed"), 1, " h/s")} />
                </dl>
                <p className="text-[11px] text-muted">
                  Release frame found by{" "}
                  {d.release_source === "user"
                    ? "you"
                    : d.release_source === "model"
                      ? "the learned model"
                      : "the frame-precise rules"}
                  . Wrong? Step to it in the viewer and press{" "}
                  <span className="text-release">Set release here</span>.
                </p>
              </>
            )}
            {tab === "Live" && (
              <>
                <LiveAngles
                  pose={frames?.get(d.clip_start + frame)}
                  arm={arm}
                  phase={phaseAt(frame, d.phases)}
                />
                <p className="text-[11px] text-muted">Updates as you play or scrub the viewer.</p>
              </>
            )}
            {tab === "Mechanics" && (
              <>
                <KineticChain metrics={d.biomechanics} />
                <MechanicsPanel metrics={d.biomechanics} compact />
              </>
            )}
            {tab === "Legality" && (
              <>
                <div className={cn("rounded-sm border p-3", verdict.cls)}>
                  <VerdictWord verdict={d.legality.verdict} />
                  <p className="mt-2 text-sm text-foreground">{d.legality.verdict_text}</p>
                </div>
                {curve("bowling_elbow") && (
                  <AngleChart
                    series={curve("bowling_elbow")!}
                    title="Bowling elbow through the delivery"
                    current={frame}
                    markers={[
                      { frame: d.phases.arm_horizontal, label: "AH" },
                      { frame: d.phases.release, label: "REL" },
                    ]}
                    onSeek={jump}
                  />
                )}
                <dl className="grid grid-cols-2 gap-2 text-sm">
                  <Kv k="Measured straightening" v={fmt(d.legality.elbow_extension_deg, 1, "°")} />
                  <Kv k="Limit" v={`${d.legality.threshold_deg.toFixed(0)}°`} />
                  <Kv k="Confidence" v={d.legality.confidence} />
                  <Kv k="Frame rate" v={`${out.analysis_fps.toFixed(0)} fps`} />
                </dl>
                <p className="text-[11px] text-muted">{d.legality.note}</p>
              </>
            )}
            {tab === "Faults" && (
              <FaultList
                faults={d.faults}
                drills={drills.data}
                notMeasured={d.not_measured}
                onJump={jump}
              />
            )}
            {tab === "Load" && (
              <>
                <div className="flex justify-center">
                  <RiskGauge score={d.injury.overall} />
                </div>
                <BodyRiskMap injury={d.injury} bowlingArm={arm} />
              </>
            )}
          </div>
        </aside>
      </div>

      <div className="space-y-4 border-t border-border p-4">
        <Section title="Action sequence" hint="Skeleton at each phase; click to open in the viewer">
          <PhaseFilmstrip d={d} frames={frames} arm={arm} edges={edges} onSeek={jump} />
        </Section>
        <Section
          title="Joint curves"
          hint="Angle and angular velocity, synced to the viewer"
          aside={
            <div className="flex gap-1">
              {JOINTS.map(([k, l]) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setJoint(k)}
                  className={cn(
                    "rounded-sm border px-2 py-0.5 text-[11px]",
                    joint === k
                      ? "border-accent text-accent"
                      : "border-border text-muted hover:text-foreground",
                  )}
                >
                  {l}
                </button>
              ))}
            </div>
          }
        >
          <div className="grid gap-4 lg:grid-cols-2">
            {curve(joint) && (
              <AngleChart
                series={curve(joint)!}
                title="Angle (°)"
                current={frame}
                markers={releaseMarkers}
                onSeek={jump}
              />
            )}
            {curve(`${joint}_velocity`) && (
              <AngleChart
                series={curve(`${joint}_velocity`)!}
                title="Angular velocity (°/s)"
                current={frame}
                markers={releaseMarkers}
                onSeek={jump}
              />
            )}
          </div>
        </Section>
        <Section title="Full mechanics" hint="Every measurement, grouped by moment of the delivery">
          <MechanicsPanel metrics={d.biomechanics} />
        </Section>
      </div>
    </div>
  );
}

function Section({
  title,
  hint,
  aside,
  children,
}: {
  title: string;
  hint?: string;
  aside?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-md border border-border bg-surface/70">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-2.5">
        <div>
          <h2 className="font-display text-sm font-semibold uppercase tracking-[0.14em]">
            {title}
          </h2>
          {hint && <p className="text-[11px] text-muted">{hint}</p>}
        </div>
        {aside}
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-baseline gap-1.5 rounded-sm border border-border bg-surface px-2 py-1 text-xs">
      <span className="text-[10px] uppercase tracking-wider text-muted">{label}</span>
      <span className="tabular font-semibold">{value}</span>
    </span>
  );
}

function Kv({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-sm border border-border bg-surface-2 px-2.5 py-1.5">
      <dt className="text-[10px] uppercase tracking-wider text-muted">{k}</dt>
      <dd className="tabular font-semibold">{v}</dd>
    </div>
  );
}

function NavBtn({ href, children }: { href: string | null; children: React.ReactNode }) {
  return href ? (
    <Link
      href={href}
      className="rounded-sm border border-border px-2 py-1 text-xs hover:border-accent"
    >
      {children}
    </Link>
  ) : (
    <span className="rounded-sm border border-border px-2 py-1 text-xs opacity-35">{children}</span>
  );
}
