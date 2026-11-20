"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { BowlerPicker } from "@/components/BowlerPicker";
import { ImageAnalysisPanel } from "@/components/ImageAnalysisPanel";
import { UploadDropzone, isImage } from "@/components/UploadDropzone";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
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
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!file) return;
    setError(null);
    setProgress(0);
    try {
      const { promise } = uploadSession(
        { file, poseFile, bowlerId: bowlerId || undefined, allowLowFps },
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
  return (
    <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
      <section className="space-y-4">
        <h1 className="text-2xl font-bold">Analyse a bowling action</h1>
        <UploadDropzone
          file={file}
          disabled={uploading}
          onFile={(f, err) => {
            setFile(f);
            setError(err ?? null);
          }}
        />
        {photo && file && <ImageAnalysisPanel key={file.name + file.size} file={file} />}
        <Card className={photo ? "hidden" : undefined}>
          <CardTitle>Bowler</CardTitle>
          <BowlerPicker value={bowlerId} onChange={setBowlerId} />
          <label className="mt-3 flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={allowLowFps}
              onChange={(e) => setAllowLowFps(e.target.checked)}
            />
            Analyse even if the video is below 60 fps (results will be less reliable)
          </label>
          <details className="mt-3 text-sm">
            <summary className="cursor-pointer text-muted">Advanced: import keypoints</summary>
            <p className="mt-2 text-muted">
              Optionally attach a PoseSequence JSON (from <code>bowlsense analyze</code> or another
              tool) to skip pose estimation.
            </p>
            <input
              type="file"
              accept=".json,application/json"
              data-testid="pose-input"
              className="mt-2 block text-sm"
              onChange={(e) => setPoseFile(e.target.files?.[0] ?? null)}
            />
          </details>
        </Card>
        {error && <ErrorBox error={error} />}
        {uploading && (
          <div aria-live="polite">
            <div className="mb-1 flex justify-between text-sm">
              <span>Uploading…</span>
              <span>{Math.round((progress ?? 0) * 100)}%</span>
            </div>
            <div
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round((progress ?? 0) * 100)}
              className="h-2 overflow-hidden rounded-full bg-surface-2"
            >
              <div
                className="h-full bg-accent transition-all"
                style={{ width: `${(progress ?? 0) * 100}%` }}
              />
            </div>
          </div>
        )}
        {!photo && (
          <Button onClick={submit} disabled={!file || uploading} className="w-full sm:w-auto">
            {uploading ? "Uploading…" : "Upload and analyse"}
          </Button>
        )}
      </section>
      <aside>
        <Card>
          <CardTitle>Recording tips</CardTitle>
          <ul className="list-disc space-y-2 pl-5 text-sm">
            {TIPS.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </Card>
      </aside>
    </div>
  );
}
