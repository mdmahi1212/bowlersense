"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { BowlerPicker } from "@/components/BowlerPicker";
import { ImageAnalysisPanel } from "@/components/ImageAnalysisPanel";
import { UploadDropzone, isImage } from "@/components/UploadDropzone";
import { Button } from "@/components/ui/Button";
import { CameraSetup } from "@/components/art/Figures";
import { PageHead } from "@/components/ui/PageHead";
import { ErrorBox } from "@/components/ui/States";
import { errorText, uploadSession } from "@/lib/api";

const TIPS = [
  "Film at 120 fps or more (60 fps is the minimum; release metrics need speed).",
  "Use a tripod, side-on to the crease, about 10 m away.",
  "Keep the whole body in frame from the run-up to the follow-through.",
  "Good light, plain background, one bowler at a time if you can.",
];

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [poseFile, setPoseFile] = useState<File | null>(null);
  const [bowlerId, setBowlerId] = useState("");
  const [allowLowFps, setAllowLowFps] = useState(false);
  const [slowMotion, setSlowMotion] = useState("auto");
  const [cameraView, setCameraView] = useState("auto");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      const { promise } = uploadSession(
        {
          file,
          poseFile,
          bowlerId: bowlerId || undefined,
          allowLowFps,
          slowMotion: slowMotion === "auto" ? undefined : Number(slowMotion),
          cameraView:
            cameraView === "auto" ? undefined : (cameraView as "side" | "front" | "behind"),
        },
        setProgress,
      );
      const out = await promise;
      router.push(`/sessions/${out.session.id}`);
    } catch (err) {
      setError(errorText(err));
      setProgress(null);
    }
  }

  const uploading = progress !== null;
  const photo = file !== null && isImage(file);
  const select =
    "w-full border-b border-foreground/60 bg-transparent py-2 outline-none focus:border-accent";
  return (
    <div className="grid gap-x-16 gap-y-10 lg:grid-cols-[minmax(0,1.45fr)_minmax(0,1fr)]">
      <section>
        <PageHead
          kicker="New analysis"
          title="Analyse a bowling action"
          deck="Upload a clip and BowlSense finds the bowler, tracks 21 joints frame by frame, pins the release and measures what the eye can't: elbow straightening, stride, alignment, load."
        />
        <UploadDropzone
          file={file}
          disabled={uploading}
          onFile={(f, err) => {
            setFile(f);
            setError(err ?? null);
          }}
        />
        {photo && file && (
          <div className="mt-8">
            <ImageAnalysisPanel key={file.name + file.size} file={file} />
          </div>
        )}
        <div className={photo ? "hidden" : "mt-10 space-y-8"}>
          <Step n="01" title="Who is bowling?">
            <BowlerPicker value={bowlerId} onChange={setBowlerId} />
          </Step>
          <Step n="02" title="About the footage">
            <div className="grid gap-6 sm:grid-cols-2">
              <label className="block text-sm">
                <span className="kicker">Slow motion</span>
                <select
                  value={slowMotion}
                  onChange={(e) => setSlowMotion(e.target.value)}
                  className={select}
                >
                  <option value="auto">Detect automatically</option>
                  <option value="1">No, real speed</option>
                  {[2, 4, 8, 10, 16].map((f) => (
                    <option key={f} value={f}>
                      Yes, {f}× slower
                    </option>
                  ))}
                </select>
                <span className="mt-1.5 block text-xs text-muted">
                  Slowed clips are analysed at their real frame rate (50 fps × 8 = 400 fps).
                </span>
              </label>
              <label className="block text-sm">
                <span className="kicker">Camera view</span>
                <select
                  value={cameraView}
                  onChange={(e) => setCameraView(e.target.value)}
                  className={select}
                >
                  <option value="auto">Detect automatically</option>
                  <option value="side">Side-on (best for legality)</option>
                  <option value="front">Front-on (facing the bowler)</option>
                  <option value="behind">From behind the bowler</option>
                </select>
              </label>
            </div>
            <label className="mt-5 flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={allowLowFps}
                onChange={(e) => setAllowLowFps(e.target.checked)}
              />
              Analyse even if the video is below 60 fps (less reliable)
            </label>
            <details className="mt-4 text-sm">
              <summary className="cursor-pointer text-muted">Advanced: import keypoints</summary>
              <p className="mt-2 text-muted">
                Attach a PoseSequence JSON (from <code>bowlsense analyze</code> or another tool) to
                skip pose estimation.
              </p>
              <input
                type="file"
                accept=".json,application/json"
                data-testid="pose-input"
                className="mt-2 block text-sm"
                onChange={(e) => setPoseFile(e.target.files?.[0] ?? null)}
              />
            </details>
          </Step>
          {error && <ErrorBox error={error} />}
          <div className="flex flex-wrap items-center gap-6 border-t border-foreground pt-6">
            <Button onClick={submit} disabled={!file || uploading} className="px-6 py-3 text-base">
              {uploading ? "Uploading…" : "Upload and analyse"}
            </Button>
            {uploading ? (
              <div className="min-w-48 flex-1" aria-live="polite">
                <div
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round((progress ?? 0) * 100)}
                  className="h-px bg-border"
                >
                  <div
                    className="h-px bg-accent transition-all"
                    style={{ width: `${(progress ?? 0) * 100}%` }}
                  />
                </div>
                <p className="tabular mt-1 text-xs text-muted">
                  {Math.round((progress ?? 0) * 100)}% uploaded
                </p>
              </div>
            ) : (
              <p className="text-sm text-muted">Nothing leaves this computer.</p>
            )}
          </div>
        </div>
        {photo && error && <ErrorBox error={error} />}
      </section>

      <aside className="lg:pt-[118px]">
        <div className="border-t border-foreground pt-3">
          <h2 className="font-display text-2xl">Filming protocol</h2>
          <p className="mt-1 text-sm text-muted">Good footage is most of the accuracy.</p>
        </div>
        <CameraSetup className="mt-6 w-full text-foreground" />
        <ol className="mt-6 divide-y divide-border border-y border-border">
          {TIPS.map((t, i) => (
            <li key={t} className="grid grid-cols-[2.5rem_1fr] py-3 text-sm leading-relaxed">
              <span className="tabular text-accent">{String(i + 1).padStart(2, "0")}</span>
              {t}
            </li>
          ))}
        </ol>
      </aside>
    </div>
  );
}

function Step({ n, title, children }: { n: string; title: string; children: React.ReactNode }) {
  return (
    <section className="grid gap-4 sm:grid-cols-[3rem_minmax(0,1fr)]">
      <span className="tabular pt-1.5 text-xs text-accent">{n}</span>
      <div>
        <h2 className="mb-4 font-display text-xl">{title}</h2>
        {children}
      </div>
    </section>
  );
}
