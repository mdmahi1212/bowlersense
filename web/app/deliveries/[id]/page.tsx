"use client";

import Link from "next/link";
import { use, useState } from "react";
import { AngleChart } from "@/components/AngleChart";
import { BodyRiskMap } from "@/components/BodyRiskMap";
import { FaultList } from "@/components/FaultList";
import { KeyframeStrip } from "@/components/KeyframeStrip";
import { RiskGauge } from "@/components/RiskGauge";
import { VideoPlayer, type SeekRequest } from "@/components/VideoPlayer";
import { Badge } from "@/components/ui/Badge";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox, Loading } from "@/components/ui/States";
import { api, fileUrl } from "@/lib/api";
import type { DeliveryOut } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { fmt } from "@/lib/utils";

const CHARTS = [
  ["elbow", "Bowling elbow angle"],
  ["front_knee", "Front knee angle"],
  ["trunk_lean", "Trunk lean from vertical"],
] as const;

export default function DeliveryPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const delivery = useAsync(() => api.delivery(id), [id]);
  if (delivery.error) return <ErrorBox error={delivery.error} />;
  if (!delivery.data) return <Loading label="Loading delivery…" />;
  return <DeliveryView out={delivery.data} />;
}

function elbowBand(out: DeliveryOut): { from: number; to: number; label: string } | undefined {
  const d = out.delivery;
  const elbow = d.angles.find((a) => a.name === "elbow");
  if (!elbow) return undefined;
  const window = elbow.values.filter(
    (v, i): v is number =>
      v !== null &&
      elbow.frames[i] >= d.phases.arm_horizontal &&
      elbow.frames[i] <= d.phases.release,
  );
  if (window.length === 0) return undefined;
  const base = Math.min(...window);
  const limit = d.legality.threshold_deg;
  return { from: base, to: base + limit, label: `${limit.toFixed(0)}° extension limit` };
}

function DeliveryView({ out }: { out: DeliveryOut }) {
  const d = out.delivery;
  const drills = useAsync(() => api.drills(), []);
  const [frame, setFrame] = useState(d.phases.release);
  const [seek, setSeek] = useState<SeekRequest | null>(null);
  const jump = (f: number) => setSeek((s) => ({ frame: f, nonce: (s?.nonce ?? 0) + 1 }));
  const markers = [
    { frame: d.phases.bfc, label: "BFC" },
    { frame: d.phases.ffc, label: "FFC" },
    { frame: d.phases.release, label: "Rel" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <Link
            href={`/sessions/${out.session_id}`}
            className="text-sm text-accent hover:underline"
          >
            ← Session
          </Link>
          <h1 className="text-2xl font-bold">
            Delivery {d.delivery_idx + 1} of {out.delivery_count}
          </h1>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge>
            {d.action_type === "unknown"
              ? "Action type: needs front view"
              : `${d.action_type.replace("_", "-")} action`}
          </Badge>
          {d.delivery_type && (
            <Badge>{d.delivery_type.delivery_type === "fast" ? "Pace" : "Spin"}</Badge>
          )}
          {!d.legality.screened ? (
            <Badge>Elbow not screened</Badge>
          ) : d.legality.flagged ? (
            <Badge className="border-bad/40 bg-bad/15 text-bad">Elbow flagged</Badge>
          ) : (
            <Badge className="border-good/40 bg-good/15 text-good">Elbow within limit</Badge>
          )}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <Card>
          <VideoPlayer
            overlaySrc={fileUrl(out.overlay)}
            rawSrc={api.videoUrl(out.session_id)}
            fps={out.fps}
            clipStart={d.clip_start}
            clipStop={d.clip_stop ?? d.clip_start + d.phases.follow_through_end + 1}
            phases={d.phases}
            seek={seek}
            onFrame={setFrame}
          />
        </Card>
        <div className="space-y-4">
          <Card>
            <CardTitle>Key numbers</CardTitle>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <dt className="text-muted">Technique score</dt>
              <dd>{d.technique_score.toFixed(0)}/100</dd>
              <dt className="text-muted">Elbow extension</dt>
              <dd>
                {d.legality.screened ? (
                  <>
                    {fmt(d.legality.elbow_extension_deg, 1, "°")} (limit {d.legality.threshold_deg}
                    °, {d.legality.confidence} confidence)
                  </>
                ) : (
                  "Not screened (see note below)"
                )}
              </dd>
              <dt className="text-muted">Stride</dt>
              <dd>{fmt(d.stride_pct, 0, "% of height")}</dd>
              <dt className="text-muted">Release height</dt>
              <dd>{fmt(d.release_height_pct, 0, "% of height")}</dd>
              <dt className="text-muted">Ball speed</dt>
              <dd>{fmt(d.speed_kmh, 0, " km/h")}</dd>
              <dt className="text-muted">Length / line</dt>
              <dd>
                {d.length ?? "—"} / {d.line ?? "—"}
              </dd>
              <dt className="text-muted">Counter-rotation</dt>
              <dd>{fmt(d.alignment?.counter_rotation_deg, 0, "°")}</dd>
              <dt className="text-muted">Release found by</dt>
              <dd>{d.release_source === "model" ? "learned model" : "rules"}</dd>
            </dl>
            {d.ball_notes.length > 0 && (
              <p className="mt-2 text-xs text-muted">{d.ball_notes.join(" ")}</p>
            )}
            <p className="mt-2 text-xs text-muted">{d.legality.note}</p>
          </Card>
          <Card className="flex justify-center">
            <RiskGauge score={d.injury.overall} />
          </Card>
        </div>
      </div>

      <Card>
        <CardTitle>Keyframes</CardTitle>
        <KeyframeStrip keyframes={d.keyframes} phases={d.phases} onJump={jump} />
      </Card>

      <Card>
        <CardTitle>Angles (synced to the video; click to seek)</CardTitle>
        <div className="grid gap-4 lg:grid-cols-3">
          {CHARTS.map(([name, title]) => {
            const series = d.angles.find((a) => a.name === name);
            return series ? (
              <AngleChart
                key={name}
                series={series}
                title={title}
                current={frame}
                markers={markers}
                band={name === "elbow" ? elbowBand(out) : undefined}
                onSeek={jump}
              />
            ) : null;
          })}
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <Card>
          <CardTitle>Technique faults and drills</CardTitle>
          {drills.error && (
            <p className="mb-2 text-sm text-bad">Couldn&apos;t load drills: {drills.error}</p>
          )}
          <FaultList
            faults={d.faults}
            drills={drills.data}
            notMeasured={d.not_measured}
            onJump={jump}
          />
        </Card>
        <Card>
          <CardTitle>Body risk map</CardTitle>
          <BodyRiskMap injury={d.injury} bowlingArm={out.bowling_arm} />
          {d.injury.top_drivers.length > 0 && (
            <div className="mt-3 text-sm">
              <p className="font-medium">Main drivers</p>
              <ul className="list-disc pl-5">
                {d.injury.top_drivers.map((t) => (
                  <li key={t}>{t}</li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      </div>

      {d.comparisons.length > 0 && (
        <Card>
          <CardTitle>Similarity to reference actions</CardTitle>
          <ul className="space-y-1 text-sm">
            {d.comparisons.map((c) => (
              <li key={c.name} className="flex justify-between">
                <span>{c.name}</span>
                <span className="text-muted">{c.similarity_pct.toFixed(0)}%</span>
              </li>
            ))}
          </ul>
          <Link
            href={`/compare?delivery=${out.delivery_id}`}
            className="mt-2 inline-block text-sm text-accent hover:underline"
          >
            Compare side by side
          </Link>
        </Card>
      )}
    </div>
  );
}
