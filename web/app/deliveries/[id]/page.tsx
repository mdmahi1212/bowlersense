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
  ReleaseSpotlight,
} from "@/components/StudioPanels";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox, Loading } from "@/components/ui/States";
import { api, errorText, fileUrl } from "@/lib/api";
import { type Arm, frameIndex } from "@/lib/pose";
import type { DeliveryOut, PoseSequence } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { cn, fmt } from "@/lib/utils";

const JOINTS = [
  ["bowling_elbow", "Bowling elbow"],
  ["front_knee", "Front knee"],
  ["back_knee", "Back knee"],
  ["trunk_lean", "Trunk lean"],
  ["upper_arm_elevation", "Upper arm"],
] as const;

export default function DeliveryPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const delivery = useAsync(() => api.delivery(id), [id]);
  const refs = useAsync(() => api.references(), []);
  if (delivery.error) return <ErrorBox error={delivery.error} />;
  if (!delivery.data || !refs.data) return <Loading label="Loading the analysis studio…" />;
  return <Studio out={delivery.data} edges={refs.data.edges} reload={delivery.reload} />;
}

function Studio({
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
        r.ok ? (r.json() as Promise<PoseSequence>) : Promise.reject(new Error(`pose ${r.status}`)),
      ),
    [out.pose],
  );
  const drills = useAsync(() => api.drills(), []);
  const frames = useMemo(() => (pose.data ? frameIndex(pose.data) : null), [pose.data]);
  const [frame, setFrame] = useState(d.phases.release);
  const [seek, setSeek] = useState<SeekRequest | null>(null);
  const [joint, setJoint] = useState<(typeof JOINTS)[number][0]>("bowling_elbow");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const jump = useCallback(
    (f: number) => setSeek((s) => ({ frame: f, nonce: (s?.nonce ?? 0) + 1 })),
    [],
  );
  const onFrame = useCallback((f: number) => setFrame(f), []);

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

  const series = (name: string) => d.biomech_curves.find((c) => c.name === name);
  const markers = [
    { frame: d.phases.bfc, label: "BFC" },
    { frame: d.phases.ffc, label: "FFC" },
    { frame: d.phases.release, label: "REL" },
  ];
  const verdict = VERDICT_STYLE[d.legality.verdict];
  const bm = new Map(d.biomechanics.map((m) => [m.key, m.value]));
  const angle = series(joint);
  const velocity = series(`${joint}_velocity`);

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-border pb-4">
        <div>
          <Link
            href={`/sessions/${out.session_id}`}
            className="font-display text-xs uppercase tracking-widest text-muted hover:text-accent"
          >
            ← Session
          </Link>
          <h1 className="font-display text-4xl font-bold uppercase tracking-wide">
            Delivery <span className="text-accent">{d.delivery_idx + 1}</span>
            <span className="text-muted">/{out.delivery_count}</span>
          </h1>
          <p className="text-sm text-muted">
            {arm}-arm · {out.view} view · {out.analysis_fps.toFixed(0)} fps real time
            {d.delivery_type
              ? ` · ${d.delivery_type.delivery_type === "fast" ? "pace" : "spin"}`
              : ""}
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <Stat label="Legality">
            <VerdictWord verdict={d.legality.verdict} />
          </Stat>
          <Stat label="Technique">
            {d.technique_score.toFixed(0)}
            <small>/100</small>
          </Stat>
          <Stat label="Injury risk">
            {d.injury.overall.toFixed(0)}
            <small> {d.injury.band}</small>
          </Stat>
          <Stat label="Arm speed">
            {fmt(bm.get("arm_angular_velocity"), 0)}
            <small>°/s</small>
          </Stat>
          <Stat label="Release height">{fmt(d.release_height_pct, 0, "%")}</Stat>
        </div>
      </header>
      {error && <ErrorBox error={error} />}
      {pose.error && <ErrorBox error={`Couldn't load the skeleton: ${pose.error}`} />}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(16rem,1fr)]">
        <Card className="p-3">
          <StudioPlayer
            videoSrc={api.videoUrl(out.session_id)}
            overlaySrc={fileUrl(out.overlay)}
            fps={out.fps}
            realFps={out.analysis_fps}
            clipStart={d.clip_start}
            clipStop={d.clip_stop ?? d.clip_start + d.phases.follow_through_end + 1}
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
          />
        </Card>
        <Card>
          <CardTitle>Live angles</CardTitle>
          <LiveAngles
            pose={frames?.get(d.clip_start + frame)}
            arm={arm}
            phase={phaseAt(frame, d.phases)}
          />
          <p
            className={cn(
              "mt-4 border-t border-border pt-3 text-xs",
              verdict.cls.split(" ").find((c) => c.startsWith("text-")),
            )}
          >
            {d.legality.verdict_text}
          </p>
          <p className="mt-1 text-[11px] text-muted">{d.legality.note}</p>
        </Card>
      </div>

      <Card>
        <CardTitle>Release spotlight</CardTitle>
        <ReleaseSpotlight
          d={d}
          pose={frames?.get(d.clip_start + d.phases.release)}
          arm={arm}
          edges={edges}
          src={api.videoUrl(out.session_id)}
          fps={out.fps}
          onOpen={() => jump(d.phases.release)}
        />
      </Card>

      <Card>
        <CardTitle>Action sequence</CardTitle>
        <PhaseFilmstrip d={d} frames={frames} arm={arm} edges={edges} onSeek={jump} />
      </Card>

      <div className="grid gap-4 xl:grid-cols-[2fr_1fr]">
        <Card>
          <CardTitle>Mechanics</CardTitle>
          <MechanicsPanel metrics={d.biomechanics} />
        </Card>
        <div className="space-y-4">
          <Card>
            <CardTitle>Kinetic chain</CardTitle>
            <KineticChain metrics={d.biomechanics} />
          </Card>
          <Card className="flex justify-center">
            <RiskGauge score={d.injury.overall} />
          </Card>
        </div>
      </div>

      <Card>
        <CardTitle
          aside={
            <div className="flex flex-wrap gap-1">
              {JOINTS.map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setJoint(key)}
                  className={cn(
                    "rounded-sm border px-2 py-1 font-display text-[11px] uppercase tracking-wider",
                    joint === key
                      ? "border-accent text-accent"
                      : "border-border text-muted hover:text-foreground",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          }
        >
          Joint curves
        </CardTitle>
        <div className="grid gap-4 lg:grid-cols-2">
          {angle && (
            <AngleChart
              series={angle}
              title="Angle (°)"
              current={frame}
              markers={markers}
              onSeek={jump}
            />
          )}
          {velocity && (
            <AngleChart
              series={velocity}
              title="Angular velocity (°/s)"
              current={frame}
              markers={markers}
              onSeek={jump}
            />
          )}
        </div>
        <p className="mt-2 text-xs text-muted">
          Synced to the player; click a chart to jump there.
        </p>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <Card>
          <CardTitle>Technique faults &amp; drills</CardTitle>
          <FaultList
            faults={d.faults}
            drills={drills.data}
            notMeasured={d.not_measured}
            onJump={jump}
          />
        </Card>
        <Card>
          <CardTitle>Body load map</CardTitle>
          <BodyRiskMap injury={d.injury} bowlingArm={arm} />
        </Card>
      </div>
    </div>
  );
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-24 rounded-sm border border-border bg-surface px-3 py-2">
      <p className="font-display text-[10px] uppercase tracking-[0.18em] text-muted">{label}</p>
      <p className="tabular text-2xl font-semibold [&_small]:text-xs [&_small]:font-normal [&_small]:text-muted">
        {children}
      </p>
    </div>
  );
}
