"use client";

import Link from "next/link";
import { use, useCallback, useMemo, useState } from "react";
import { AngleChart } from "@/components/AngleChart";
import { BodyRiskMap } from "@/components/BodyRiskMap";
import { FaultList } from "@/components/FaultList";
import { VerdictWord } from "@/components/LegalityVerdict";
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
import { SectionHead } from "@/components/ui/SectionHead";
import { cleanName, cn, fmt } from "@/lib/utils";

type Tab = "Legality" | "Joint angles" | "Mechanics" | "Faults" | "Injury load" | "Sequence";
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
  const [tab, setTab] = useState<Tab>("Legality");
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
  const src = api.videoUrl(out.session_id);
  const releaseMarkers = [
    { frame: d.phases.ffc, label: "FFC" },
    { frame: d.phases.release, label: "REL" },
  ];

  const tabs: Tab[] = [
    "Legality",
    "Joint angles",
    "Mechanics",
    "Faults",
    "Injury load",
    "Sequence",
  ];

  return (
    <div className="mx-auto max-w-[1440px] space-y-10 pb-10">
      <header className="flex flex-wrap items-end justify-between gap-4 pt-2">
        <div className="min-w-0">
          <Link
            href={`/sessions/${out.session_id}`}
            className="block max-w-md truncate text-sm text-muted hover:text-foreground"
          >
            ← {cleanName(session.data?.session.original_name)}
          </Link>
          <h1 className="mt-1 text-2xl font-semibold">
            Delivery {d.delivery_idx + 1} of {out.delivery_count}
          </h1>
          <p className="mt-1 text-sm text-muted">
            Release at frame {d.phases.release} (
            {d.release_source === "user" ? "set by you" : "auto-detected"}) ·{" "}
            {out.analysis_fps.toFixed(0)} fps real time · {arm}-arm
          </p>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <VerdictWord verdict={d.legality.verdict} verified={d.legality.screened} />
          <span className="flex gap-3">
            <NavBtn href={pos > 0 ? `/deliveries/${ids[pos - 1]}` : null}>← Prev</NavBtn>
            <NavBtn href={pos >= 0 && pos < ids.length - 1 ? `/deliveries/${ids[pos + 1]}` : null}>
              Next →
            </NavBtn>
          </span>
          <a
            href={api.reportUrl(out.session_id)}
            download
            className="rounded-md border border-border px-3 py-1.5 hover:border-accent"
          >
            PDF
          </a>
        </div>
      </header>
      {(error || pose.error) && (
        <ErrorBox error={error ?? `Couldn't load the skeleton: ${pose.error}`} />
      )}

      <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_380px]">
        <section className="min-w-0 space-y-4" aria-label="Viewer">
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

        <aside className="space-y-5" aria-label="At release">
          <SectionHead title="At release" hint="Slow-motion loop around the release point" />
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
          <div className="grid grid-cols-2 gap-x-3 gap-y-4">
            <AngleDial label="Elbow at release" value={rel.elbow} band={[160, 180]} highlight />
            <AngleDial label="Front knee" value={rel.frontKnee} band={[160, 180]} />
            <AngleDial label="Trunk lean" value={rel.trunkLean} band={[0, 35]} />
            <AngleDial
              label="Arm elevation"
              value={rel.upperArm === null ? null : rel.upperArm + 90}
              band={[150, 180]}
            />
          </div>
          <dl className="divide-y divide-border border-y border-border text-sm">
            <Kv k="Elbow straightening" v={fmt(d.legality.elbow_extension_deg, 1, "°")} />
            <Kv k="Release height" v={fmt(d.release_height_pct, 0, "% of height")} />
            <Kv k="Arm speed" v={fmt(bm.get("arm_angular_velocity"), 0, "°/s")} />
            <Kv k="Technique" v={`${d.technique_score.toFixed(0)} / 100`} />
          </dl>
          <p className="text-xs leading-relaxed text-muted">
            Wrong release frame? Step to it in the viewer and press{" "}
            <span className="text-release">Set release here</span>.
          </p>
        </aside>
      </div>

      <section>
        <div className="mb-6 flex gap-6 overflow-x-auto border-b border-border" role="tablist">
          {tabs.map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={cn(
                "-mb-px whitespace-nowrap border-b-2 pb-2.5 text-sm",
                tab === t
                  ? "border-accent text-foreground"
                  : "border-transparent text-muted hover:text-foreground",
              )}
            >
              {t}
            </button>
          ))}
        </div>

        {tab === "Legality" && (
          <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
            <div className="space-y-4">
              <VerdictWord verdict={d.legality.verdict} verified={d.legality.screened} />
              <p className="text-[15px] leading-relaxed">
                {d.legality.verdict_text.split(" Unverified:")[0]}
              </p>
              {!d.legality.screened && d.legality.not_screened_reason && (
                <p className="border-l-2 border-warn/60 pl-3 text-sm text-muted">
                  <span className="text-warn">Why it&apos;s not certain: </span>
                  {d.legality.not_screened_reason.replace(/^Unverified:\s*/, "")} The verdict is
                  read straight from the angle, so treat it as a guide and re-film side-on at 120+
                  fps to confirm.
                </p>
              )}
              <dl className="divide-y divide-border border-y border-border text-sm">
                <Kv k="Measured straightening" v={fmt(d.legality.elbow_extension_deg, 1, "°")} />
                <Kv k="Limit" v={`${d.legality.threshold_deg.toFixed(0)}°`} />
                <Kv k="Confidence" v={d.legality.confidence} />
              </dl>
              <p className="text-xs text-muted">{d.legality.note.split(" Unverified:")[0]}</p>
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
          </div>
        )}
        {tab === "Joint angles" && (
          <div className="grid gap-10 lg:grid-cols-[280px_minmax(0,1fr)]">
            <div>
              <p className="mb-3 text-sm text-muted">At the current frame</p>
              <LiveAngles
                pose={frames?.get(d.clip_start + frame)}
                arm={arm}
                phase={phaseAt(frame, d.phases)}
              />
            </div>
            <div>
              <div className="mb-4 flex flex-wrap gap-2">
                {JOINTS.map(([k, l]) => (
                  <button
                    key={k}
                    type="button"
                    onClick={() => setJoint(k)}
                    className={cn(
                      "rounded-full border px-3 py-1 text-xs",
                      joint === k
                        ? "border-accent text-accent"
                        : "border-border text-muted hover:text-foreground",
                    )}
                  >
                    {l}
                  </button>
                ))}
              </div>
              <div className="grid gap-6 lg:grid-cols-2">
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
            </div>
          </div>
        )}
        {tab === "Mechanics" && (
          <div className="space-y-8">
            <KineticChain metrics={d.biomechanics} />
            <MechanicsPanel metrics={d.biomechanics} />
          </div>
        )}
        {tab === "Faults" && (
          <div className="max-w-3xl">
            <FaultList
              faults={d.faults}
              drills={drills.data}
              notMeasured={d.not_measured}
              onJump={jump}
            />
          </div>
        )}
        {tab === "Injury load" && (
          <div className="grid items-start gap-10 md:grid-cols-[260px_minmax(0,1fr)]">
            <RiskGauge score={d.injury.overall} />
            <BodyRiskMap injury={d.injury} bowlingArm={arm} />
          </div>
        )}
        {tab === "Sequence" && (
          <PhaseFilmstrip d={d} frames={frames} arm={arm} edges={edges} onSeek={jump} />
        )}
      </section>
    </div>
  );
}

function Kv({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4 py-2">
      <dt className="text-muted">{k}</dt>
      <dd className="tabular">{v}</dd>
    </div>
  );
}

function NavBtn({ href, children }: { href: string | null; children: React.ReactNode }) {
  return href ? (
    <Link href={href} className="text-muted hover:text-foreground">
      {children}
    </Link>
  ) : (
    <span className="text-muted opacity-35">{children}</span>
  );
}
