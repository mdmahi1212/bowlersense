"use client";

/* eslint-disable @next/next/no-img-element -- local API image, no optimisation server */
import { useState } from "react";
import { analyzeImage, errorText, fileUrl } from "@/lib/api";
import type { ImageAnalysis } from "@/lib/types";
import { Badge } from "./ui/Badge";
import { Button } from "./ui/Button";
import { Card, CardTitle } from "./ui/Card";
import { ErrorBox } from "./ui/States";

const PHASES = [
  ["auto", "Auto (best guess)"],
  ["bfc", "Back-foot contact"],
  ["ffc", "Front-foot contact"],
  ["release", "Release"],
] as const;

/** Image mode: one photo, the phase it shows, and the posture checks that work on one frame. */
export function ImageAnalysisPanel({
  file,
  bowlingArm,
}: {
  file: File;
  bowlingArm?: "right" | "left";
}) {
  const [phase, setPhase] = useState<(typeof PHASES)[number][0]>("auto");
  const [view, setView] = useState<"auto" | "side" | "front">("auto");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImageAnalysis | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await analyzeImage({ file, phase, view, bowlingArm }));
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  const select = "border border-border bg-surface-2 px-3 py-2";
  return (
    <div className="space-y-4">
      <Card>
        <CardTitle>Photo analysis</CardTitle>
        <div className="flex flex-wrap items-end gap-3 text-sm">
          <label className="flex flex-col gap-1">
            Which moment does the photo show?
            <select
              value={phase}
              onChange={(e) => setPhase(e.target.value as typeof phase)}
              className={select}
            >
              {PHASES.map(([v, label]) => (
                <option key={v} value={v}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            Camera view
            <select
              value={view}
              onChange={(e) => setView(e.target.value as typeof view)}
              className={select}
            >
              <option value="auto">Auto</option>
              <option value="side">Side-on</option>
              <option value="front">Front-on</option>
            </select>
          </label>
          <Button onClick={run} disabled={busy}>
            {busy ? "Analysing…" : "Analyse photo"}
          </Button>
        </div>
        <p className="mt-2 text-xs text-muted">
          A photo shows posture at one moment. Elbow legality, speeds and timing need a video.
        </p>
      </Card>
      {error && <ErrorBox error={error} />}
      {result && (
        <Card data-testid="image-result">
          <CardTitle
            aside={
              <div className="flex gap-2">
                <Badge>
                  {result.phase.toUpperCase()} ({result.phase_source})
                </Badge>
                <Badge>
                  {result.view} view · {result.bowling_arm}-arm
                </Badge>
              </div>
            }
          >
            Posture checks
          </CardTitle>
          <div className="grid gap-4 lg:grid-cols-2">
            {result.annotated && (
              <img
                src={fileUrl(result.annotated)}
                alt="Photo with the detected skeleton"
                className="w-full "
              />
            )}
            <ul className="space-y-3 text-sm">
              {result.checks.map((c) => (
                <li key={c.code} className="border border-border p-3">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold">{c.name}</span>
                    {c.flagged === true && (
                      <Badge className="border-bad/40 bg-bad/15 text-bad">Check this</Badge>
                    )}
                    {c.flagged === false && (
                      <Badge className="border-good/40 bg-good/15 text-good">OK</Badge>
                    )}
                  </div>
                  <p className="mt-1">
                    {c.value === null
                      ? "Not visible in the photo"
                      : `${c.value.toFixed(0)} ${c.unit}`}
                    {c.threshold !== null && ` (limit ${c.threshold.toFixed(0)})`}
                  </p>
                  <p className="mt-1 text-muted">{c.note}</p>
                </li>
              ))}
              {result.warnings.map((w) => (
                <li key={w} className="text-warn">
                  {w}
                </li>
              ))}
              <li className="text-xs text-muted">{result.note}</li>
            </ul>
          </div>
        </Card>
      )}
    </div>
  );
}
