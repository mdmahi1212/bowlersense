"use client";

import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { Card, CardTitle } from "@/components/ui/Card";
import { ErrorBox, Loading } from "@/components/ui/States";
import { api, errorText } from "@/lib/api";
import type { LocalSettings } from "@/lib/types";
import { useAsync } from "@/lib/useAsync";
import { fmtBytes } from "@/lib/utils";

export default function SettingsPage() {
  const settings = useAsync(() => api.settings(), []);
  const models = useAsync(() => api.llmModels(), []);
  const health = useAsync(() => api.health(), []);
  if (settings.error)
    return (
      <ErrorBox error={settings.error} action={<Button onClick={settings.reload}>Retry</Button>} />
    );
  if (!settings.data) return <Loading />;
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Settings</h1>
      <SettingsForm
        key={JSON.stringify(settings.data.settings)}
        initial={settings.data.settings}
        modelNames={models.data?.models ?? []}
        ollamaUp={models.data?.available ?? false}
        ollamaDetail={models.data?.detail ?? models.error ?? null}
        gpu={settings.data.gpu}
        onSaved={settings.reload}
      />
      <Card>
        <CardTitle>Models on this computer</CardTitle>
        {health.data ? (
          <ul className="grid gap-1 text-sm sm:grid-cols-2">
            {Object.entries(health.data.models).map(([name, ok]) => (
              <li key={name}>
                <span className={ok ? "text-good" : "text-warn"}>
                  {ok ? "✓ present" : "✗ missing"}
                </span>{" "}
                {name}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted">{health.error ?? "Checking…"}</p>
        )}
        <p className="mt-2 text-xs text-muted">
          Missing files: run python scripts/download_models.py once.
        </p>
      </Card>
      <DangerZone storageBytes={settings.data.storage_bytes} onDeleted={settings.reload} />
    </div>
  );
}

function SettingsForm({
  initial,
  modelNames,
  ollamaUp,
  ollamaDetail,
  gpu,
  onSaved,
}: {
  initial: LocalSettings;
  modelNames: string[];
  ollamaUp: boolean;
  ollamaDetail: string | null;
  gpu: Record<string, boolean>;
  onSaved: () => void;
}) {
  const [s, setS] = useState<LocalSettings>(initial);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const hasGpu = gpu.torch_cuda || gpu.onnxruntime_cuda;
  const options = modelNames.includes(s.coach_model) ? modelNames : [s.coach_model, ...modelNames];

  async function save() {
    setError(null);
    try {
      await api.saveSettings(s);
      setStatus("Saved. New analyses use these settings.");
      onSaved();
    } catch (err) {
      setError(errorText(err));
    }
  }

  const field = "rounded-md border border-border bg-surface-2 px-3 py-2";
  return (
    <Card>
      <CardTitle>Analysis</CardTitle>
      <div className="grid gap-4 text-sm sm:grid-cols-2">
        <label className="flex flex-col gap-1">
          Coach model (local Ollama)
          <select
            value={s.coach_model}
            onChange={(e) => setS({ ...s, coach_model: e.target.value })}
            className={field}
          >
            {options.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <span className="text-xs text-muted">
            {ollamaUp
              ? `${modelNames.length} model(s) pulled in Ollama.`
              : `Ollama isn't running (${ollamaDetail ?? "not reachable"}); the rule-based coach is used.`}
          </span>
        </label>
        <label className="flex flex-col gap-1">
          Pose backend
          <select
            value={s.pose_backend}
            onChange={(e) =>
              setS({ ...s, pose_backend: e.target.value as LocalSettings["pose_backend"] })
            }
            className={field}
          >
            <option value="rtm">RTMPose (recommended)</option>
            <option value="mediapipe">MediaPipe (fallback)</option>
          </select>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={s.use_gpu}
            onChange={(e) => setS({ ...s, use_gpu: e.target.checked })}
          />
          Use the GPU when available {hasGpu ? "(CUDA found)" : "(no CUDA GPU found: CPU is used)"}
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={s.use_phase_model}
            onChange={(e) => setS({ ...s, use_phase_model: e.target.checked })}
          />
          Use the learned release model (trained on real footage; better on match video)
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={s.use_llm}
            onChange={(e) => setS({ ...s, use_llm: e.target.checked })}
          />
          Use the local LLM coach (otherwise the rule-based report)
        </label>
      </div>
      <div className="mt-4 flex items-center gap-3">
        <Button onClick={save}>Save settings</Button>
        {status && <span className="text-sm text-good">{status}</span>}
        {error && <span className="text-sm text-bad">{error}</span>}
      </div>
    </Card>
  );
}

function DangerZone({ storageBytes, onDeleted }: { storageBytes: number; onDeleted: () => void }) {
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  async function wipe() {
    setError(null);
    try {
      await api.deleteAllData();
      setDone(true);
      setConfirm("");
      onDeleted();
    } catch (err) {
      setError(errorText(err));
    }
  }

  return (
    <Card className="border-bad/40">
      <CardTitle>Your data</CardTitle>
      <p className="text-sm">
        Videos, results and reports use <strong>{fmtBytes(storageBytes)}</strong> on this computer.
        Nothing is uploaded anywhere.
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
        <label htmlFor="confirm-delete">
          Type DELETE to remove every session, bowler and file:
        </label>
        <input
          id="confirm-delete"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          className="w-28 rounded-md border border-border bg-surface-2 px-2 py-1.5"
        />
        <Button variant="danger" disabled={confirm !== "DELETE"} onClick={wipe}>
          Delete all my data
        </Button>
      </div>
      {done && <p className="mt-2 text-sm text-good">All local data deleted.</p>}
      {error && <p className="mt-2 text-sm text-bad">{error}</p>}
    </Card>
  );
}
