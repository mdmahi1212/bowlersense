"use client";

import Link from "next/link";
import { use, useCallback, useMemo, useState } from "react";
import { AngleChart } from "@/components/AngleChart";
import { CoachChat } from "@/components/CoachChat";
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
import { BiomechLayer, type LayerFlags } from "@/components/studio/BiomechLayer";
import { FaultFocus, faultJoints } from "@/components/studio/FaultFocus";
import { MeasureLayer, type MeasureMode, describe } from "@/components/studio/MeasureLayer";
import { RhythmPanel } from "@/components/studio/RhythmPanel";
import { bodyHeightPx, headStability, hipMid, wristPath } from "@/lib/biomech";
import { ReleaseLoop } from "@/components/studio/ReleaseLoop";
import { type Lane, TrackDock } from "@/components/studio/TrackDock";
import { ErrorBox, Loading } from "@/components/ui/States";
import { api, errorText, fileUrl } from "@/lib/api";
import { type Arm, frameAngles, frameIndex } from "@/lib/pose";
import type { Annotation, DeliveryOut, PoseSequence, ReferenceAction } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { PageHead } from "@/components/ui/PageHead";
import { cleanName, cn, fmt } from "@/lib/utils";

type Tab =
  | "Legality"
  | "Joint angles"
  | "Mechanics"
  | "Faults"
  | "Injury load"
  | "Sequence"
  | "Measurements"
  | "Ask the coach";
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
    <Workspace
      key={id}
      out={delivery.data}
      edges={refs.data.edges}
      reference={refs.data.references[0] ?? null}
      reload={delivery.reload}
    />
  );
}

function Workspace({
  out,
  edges,
  reference,
  reload,
}: {
  out: DeliveryOut;
  edges: string[][];
  reference: ReferenceAction | null;
  reload: () => void;
}) {
  const d = out.delivery;
  const arm = out.bowling_arm as Arm;
  const [layers, setLayers] = useState<LayerFlags>({ speed: false, com: false, arm: false });
  const [focus, setFocus] = useState<string[]>([]);
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
  const heightPx = useMemo(() => (frames ? bodyHeightPx(frames) : null), [frames]);
  const ref = useMemo(
    () =>
      reference
        ? { name: reference.name, frames: frameIndex(reference.sequence), phases: reference.phases }
        : null,
    [reference],
  );
  const refPath = useMemo(() => {
    if (!ref || !frames || !heightPx) return null;
    const relU = d.clip_start + d.phases.release;
    const hu = hipMid(frames.get(relU));
    const hr = hipMid(ref.frames.get(ref.phases.release));
    const hRef = bodyHeightPx(ref.frames);
    const bu = hipMid(frames.get(d.clip_start + d.phases.bfc));
    const br = hipMid(ref.frames.get(ref.phases.bfc));
    if (!hu || !hr || !hRef || !bu || !br) return null;
    const flip = Math.sign(hu.x - bu.x) * Math.sign(hr.x - br.x) < 0 ? -1 : 1;
    const k = heightPx / hRef;
    const refArm = (reference?.sequence.bowling_arm ?? arm) as Arm;
    return wristPath(ref.frames, refArm, ref.phases.bfc, ref.phases.follow_through_end).map(
      (p) => ({
        x: hu.x + flip * (p.x - hr.x) * k,
        y: hu.y + (p.y - hr.y) * k,
      }),
    );
  }, [ref, frames, heightPx, d, arm, reference]);
  const live = frameAngles(frames?.get(d.clip_start + frame), arm);
  const frameContext =
    `[Delivery ${d.delivery_idx + 1}, frame ${frame} (${phaseAt(frame, d.phases)}), ` +
    `${(((frame - d.phases.release) / out.analysis_fps) * 1000).toFixed(0)} ms from release. ` +
    `Angles at this frame: elbow ${fmt(live.elbow, 0)} deg, front knee ${fmt(live.frontKnee, 0)} deg, ` +
    `trunk lean ${fmt(live.trunkLean, 0)} deg, upper arm ${fmt(live.upperArm, 0)} deg. ` +
    `Faults on this delivery: ${d.faults.map((f) => f.name).join(", ") || "none"}.]`;
  const head = useMemo(
    () =>
      frames && heightPx
        ? headStability(frames, d.clip_start + d.phases.release, out.analysis_fps, heightPx)
        : null,
    [frames, heightPx, d, out.analysis_fps],
  );
  const [mode, setMode] = useState<MeasureMode>("off");
  const saved = useAsync(() => api.annotations(out.delivery_id), [out.delivery_id]);
  const [marks, setMarks] = useState<Annotation[] | null>(null);
  const notes = marks ?? saved.data?.annotations ?? [];
  function saveMarks(next: Annotation[]) {
    setMarks(next);
    api.saveAnnotations(out.delivery_id, next).catch((err) => setError(errorText(err)));
  }
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
    { label: "Elbow", unit: "°", values: values("bowling_elbow"), colour: "#b3261e" },
    { label: "Front knee", unit: "°", values: values("front_knee"), colour: "#16140f" },
    { label: "Trunk lean", unit: "°", values: values("trunk_lean"), colour: "#2f5d8a" },
    {
      label: "Arm speed",
      unit: "°/s",
      values: values("upper_arm_elevation_velocity", true),
      colour: "#a8670d",
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
    "Measurements",
    "Ask the coach",
  ];

  return (
    <div className="space-y-12">
      <PageHead
        kicker={
          <>
            <Link href={`/sessions/${out.session_id}`} className="hover:text-foreground">
              {cleanName(session.data?.session.original_name)}
            </Link>{" "}
            / Delivery report
          </>
        }
        title={`Delivery ${d.delivery_idx + 1} of ${out.delivery_count}`}
        deck={`Release at frame ${d.phases.release}, ${d.release_source === "user" ? "set by you" : "auto-detected"}. ${out.analysis_fps.toFixed(0)} fps real time, ${arm}-arm.`}
        aside={
          <>
            <VerdictWord verdict={d.legality.verdict} size="sm" verified={d.legality.screened} />
            <span className="h-4 w-px bg-border" />
            <NavBtn href={pos > 0 ? `/deliveries/${ids[pos - 1]}` : null}>← Previous</NavBtn>
            <NavBtn href={pos >= 0 && pos < ids.length - 1 ? `/deliveries/${ids[pos + 1]}` : null}>
              Next →
            </NavBtn>
            <Link
              href={`/deliveries/${out.delivery_id}/vs`}
              className="border border-foreground px-4 py-2 hover:bg-foreground hover:text-background"
            >
              Side by side
            </Link>
            <a
              href={api.reportUrl(out.session_id)}
              download
              className="bg-foreground px-4 py-2 text-background hover:bg-accent"
            >
              PDF report
            </a>
          </>
        }
      />
      {(error || pose.error) && (
        <ErrorBox error={error ?? `Couldn't load the skeleton: ${pose.error}`} />
      )}

      <div data-reveal className="rule-draw grid gap-10 pt-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <section className="min-w-0 space-y-4" aria-label="Viewer">
          <div className="flex items-baseline justify-between">
            <div>
              <p className="kicker">Plate 1</p>
              <h2 className="mt-1 font-display text-2xl">The delivery, frame by frame</h2>
            </div>
            <div
              className="flex flex-wrap items-center gap-1 text-sm"
              role="group"
              aria-label="Analysis layers"
            >
              <span className="kicker mr-2">Show</span>
              {(
                [
                  ["speed", "Joint speed"],
                  ["com", "Centre of mass"],
                  ["arm", "Arm path"],
                ] as const
              ).map(([k, label]) => (
                <button
                  key={k}
                  type="button"
                  aria-pressed={layers[k]}
                  onClick={() => setLayers((l) => ({ ...l, [k]: !l[k] }))}
                  className={cn(
                    "px-2.5 py-1 transition-colors",
                    layers[k] ? "bg-accent text-white" : "text-muted hover:text-foreground",
                  )}
                >
                  {label}
                </button>
              ))}
              {focus.length > 0 && (
                <button
                  type="button"
                  onClick={() => setFocus([])}
                  className="ml-1 px-2 py-1 text-xs text-accent"
                >
                  Clear highlight ×
                </button>
              )}
            </div>
            <div className="flex items-center gap-1 text-sm" role="group" aria-label="Measure">
              <span className="kicker mr-2">Measure</span>
              {(["off", "line", "angle", "note"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  aria-pressed={mode === m}
                  onClick={() => setMode(m)}
                  className={cn(
                    "px-2.5 py-1 capitalize transition-colors",
                    mode === m
                      ? "bg-foreground text-background"
                      : "text-muted hover:text-foreground",
                  )}
                >
                  {m}
                </button>
              ))}
              {mode !== "off" && (
                <span className="ml-2 hidden text-xs text-muted lg:inline">
                  {mode === "angle"
                    ? "Click 3 points, vertex second"
                    : mode === "note"
                      ? "Click where the note belongs"
                      : "Click 2 points"}{" "}
                  · right-click cancels
                </span>
              )}
            </div>
          </div>
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
            overlay={
              <>
                <BiomechLayer
                  width={out.width}
                  height={out.height}
                  frames={frames}
                  frame={d.clip_start + frame}
                  arm={arm}
                  fps={out.analysis_fps}
                  heightPx={heightPx}
                  window={[d.clip_start + d.phases.bfc, d.clip_start + d.phases.follow_through_end]}
                  release={d.clip_start + d.phases.release}
                  show={layers}
                  focus={focus}
                  refPath={layers.arm ? refPath : null}
                />
                <MeasureLayer
                  width={out.width}
                  height={out.height}
                  frame={frame}
                  annotations={notes}
                  mode={mode}
                  onAdd={(a) => {
                    saveMarks([...notes, a]);
                    setTab("Measurements");
                  }}
                />
              </>
            }
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
          <div>
            <p className="kicker">Plate 2</p>
            <h2 className="mt-1 font-display text-2xl">At the release point</h2>
            <p className="text-sm text-muted">Slow-motion loop, ±120 ms around release.</p>
          </div>
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
          <div className="flex gap-2 text-sm">
            <a
              href={api.clipUrl(out.delivery_id, "mp4")}
              download
              className="flex-1 bg-foreground px-3 py-2 text-center text-background transition-colors hover:bg-accent"
            >
              Export slow-mo MP4
            </a>
            <a
              href={api.clipUrl(out.delivery_id, "gif")}
              download
              className="border border-foreground px-3 py-2 text-center transition-colors hover:bg-foreground hover:text-background"
            >
              GIF
            </a>
            <a
              href={api.posterUrl(out.delivery_id)}
              download
              className="border border-foreground px-3 py-2 text-center transition-colors hover:bg-foreground hover:text-background"
              title="Shareable image of the release frame with the verdict and key numbers"
            >
              Poster
            </a>
          </div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-3">
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
            <Kv
              k="Ball speed"
              v={
                d.speed_kmh != null
                  ? `${d.speed_kmh.toFixed(0)} km/h`
                  : d.speed_estimate_kmh != null
                    ? `≈ ${d.speed_estimate_kmh.toFixed(0)} km/h (rough)`
                    : "—"
              }
            />
            <Kv
              k="Head stability"
              v={head ? `${head.score.toFixed(0)} / 100 · moves ${head.movePct.toFixed(1)}%` : "—"}
            />
          </dl>
          <p className="text-xs leading-relaxed text-muted">
            Wrong release frame? Step to it in the viewer and press{" "}
            <span className="text-release">Set release here</span>.
          </p>
        </aside>
      </div>

      <section>
        <div className="mb-8 flex gap-8 overflow-x-auto border-b border-foreground" role="tablist">
          {tabs.map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={cn(
                "uline -mb-px whitespace-nowrap pb-2 font-display text-lg transition-colors",
                tab === t ? "text-foreground" : "text-muted hover:text-foreground",
              )}
            >
              {t}
            </button>
          ))}
        </div>

        <div key={tab} className="page-enter">
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
                  <Kv
                    k="Front-foot no-ball"
                    v={
                      d.legality.front_foot_no_ball != null
                        ? d.legality.front_foot_no_ball
                          ? "No-ball (calibrated crease)"
                          : "Behind the line (calibrated)"
                        : d.no_ball?.status === "no_ball"
                          ? `No-ball? heel ${d.no_ball.margin_cm?.toFixed(0)} cm over`
                          : d.no_ball?.status === "legal"
                            ? `Behind the line (${Math.abs(d.no_ball.margin_cm ?? 0).toFixed(0)} cm)`
                            : "crease not found"
                    }
                  />
                  <Kv k="Limit" v={`${d.legality.threshold_deg.toFixed(0)}°`} />
                  <Kv k="Confidence" v={d.legality.confidence} />
                </dl>
                {d.no_ball && d.no_ball.status !== "not_found" && (
                  <button
                    type="button"
                    onClick={() => jump(d.phases.ffc)}
                    className="text-sm text-accent hover:underline"
                  >
                    Check the front-foot-contact frame →
                  </button>
                )}
                <p className="text-xs text-muted">
                  {d.legality.note.split(" Unverified:")[0]}
                  {d.no_ball ? ` No-ball: ${d.no_ball.note}` : ""}
                </p>
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
            <div className="space-y-12">
              {frames && heightPx && (
                <RhythmPanel
                  frames={frames}
                  clipStart={d.clip_start}
                  phases={d.phases}
                  fps={out.analysis_fps}
                  heightPx={heightPx}
                  onSeek={jump}
                />
              )}
              <KineticChain metrics={d.biomechanics} />
              <MechanicsPanel metrics={d.biomechanics} />
            </div>
          )}
          {tab === "Faults" && (
            <div className="space-y-10">
              <FaultFocus
                faults={d.faults}
                frames={frames}
                clipStart={d.clip_start}
                phases={d.phases}
                arm={arm}
                edges={edges}
                reference={ref}
                onShow={(f, code) => {
                  jump(f);
                  setFocus(faultJoints(code, arm));
                  window.scrollTo({ top: 0, behavior: "smooth" });
                }}
              />
              <div className="max-w-3xl">
                <FaultList
                  faults={d.faults}
                  drills={drills.data}
                  notMeasured={d.not_measured}
                  onJump={(f, code) => {
                    jump(f);
                    if (code) setFocus(faultJoints(code, arm));
                  }}
                />
              </div>
            </div>
          )}
          {tab === "Injury load" && (
            <div className="grid items-start gap-10 md:grid-cols-[260px_minmax(0,1fr)]">
              <RiskGauge score={d.injury.overall} />
              <BodyRiskMap injury={d.injury} bowlingArm={arm} />
            </div>
          )}
          {tab === "Measurements" && (
            <div className="max-w-3xl">
              {notes.length === 0 ? (
                <p className="text-muted">
                  No measurements yet. Pause on a frame, choose <b>Line</b> or <b>Angle</b> above
                  the video and click on the body. They are saved with this delivery.
                </p>
              ) : (
                <ol className="divide-y divide-border border-y border-border">
                  {notes.map((a, i) => (
                    <li
                      key={i}
                      className="grid grid-cols-[2rem_6rem_minmax(0,1fr)_auto_auto] items-center gap-4 py-3"
                    >
                      <span className="tabular text-xs text-accent">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="capitalize text-muted">{a.kind}</span>
                      <span className="tabular text-lg">{describe(a)}</span>
                      <button
                        type="button"
                        onClick={() => jump(a.frame)}
                        className="text-sm text-muted hover:text-foreground"
                      >
                        Frame {a.frame} →
                      </button>
                      <button
                        type="button"
                        onClick={() => saveMarks(notes.filter((_, j) => j !== i))}
                        className="text-sm text-muted hover:text-bad"
                        aria-label={`Delete measurement ${i + 1}`}
                      >
                        Delete
                      </button>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          )}
          {tab === "Ask the coach" && (
            <div className="grid gap-10 lg:grid-cols-[300px_minmax(0,1fr)]">
              <div>
                <p className="kicker">The coach sees this frame</p>
                <p className="mt-2 font-display text-2xl">
                  Frame {frame} · {phaseAt(frame, d.phases)}
                </p>
                <dl className="mt-3 divide-y divide-border border-y border-border text-sm">
                  <Kv k="Elbow" v={fmt(live.elbow, 0, "°")} />
                  <Kv k="Front knee" v={fmt(live.frontKnee, 0, "°")} />
                  <Kv k="Trunk lean" v={fmt(live.trunkLean, 0, "°")} />
                  <Kv k="Upper arm" v={fmt(live.upperArm, 0, "°")} />
                </dl>
                <p className="mt-3 text-xs text-muted">
                  Pause on any frame and ask, e.g. &ldquo;What is wrong with my front leg
                  here?&rdquo; Only these numbers are sent to your local coach model, never the
                  video.
                </p>
              </div>
              <CoachChat
                sessionId={out.session_id}
                context={frameContext}
                placeholder="Ask about this frame…"
              />
            </div>
          )}
          {tab === "Sequence" && (
            <PhaseFilmstrip d={d} frames={frames} arm={arm} edges={edges} onSeek={jump} />
          )}
        </div>
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
