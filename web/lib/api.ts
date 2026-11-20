/**
 * Typed client for the local BowlSense API (127.0.0.1 only). Types come from lib/types.ts,
 * which is generated from the Python pydantic models.
 */
import type {
  Annotation,
  AnnotationsIn,
  ProgressOut,
  BowlerIn,
  BowlerRecord,
  CalibrationIn,
  ChatTurn,
  CoachOut,
  DeliveryOut,
  EvaluationReport,
  ImageAnalysis,
  Drill,
  LocalSettings,
  PeopleTracks,
  ReferencesOut,
  SessionOut,
  SessionRecord,
  SettingsOut,
  WorkloadOut,
} from "./types";

const DEFAULT_API = "http://127.0.0.1:8000";

/** The local API address: from the page (set by the server at run time), else the default. */
export function apiBase(): string {
  if (typeof document !== "undefined") {
    const meta = document.querySelector<HTMLMetaElement>('meta[name="bowlsense-api"]');
    if (meta?.content) return meta.content.replace(/\/$/, "");
  }
  return DEFAULT_API;
}

/** An API call that failed, with the server's message when it sent one. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function errorFrom(res: Response): Promise<ApiError> {
  let message = `${res.status} ${res.statusText}`;
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      message = typeof detail === "string" ? detail : JSON.stringify(detail);
    }
  } catch {
    // not JSON: keep the status text
  }
  return new ApiError(res.status, message);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${apiBase()}${path}`, {
      ...init,
      headers: init?.body ? { "Content-Type": "application/json", ...init.headers } : init?.headers,
    });
  } catch {
    throw new ApiError(
      0,
      `Can't reach the BowlSense API at ${apiBase()}. Is \`python run.py\` running?`,
    );
  }
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({ body: JSON.stringify(body) });

/** URL of a stored output file (overlay, keyframe, PDF). */
export const fileUrl = (storagePath: string): string =>
  `${apiBase()}/files/${storagePath.split("/").map(encodeURIComponent).join("/")}`;

export interface Health {
  status: string;
  version: string;
  models: Record<string, boolean>;
  gpu: Record<string, boolean>;
}

export interface LlmModels {
  available: boolean;
  models: string[];
  selected: string;
  detail?: string;
}

export interface SessionEvent {
  status: SessionRecord["status"] | "deleted";
  progress: number;
  stage: string | null;
  attempts: number;
  error: string | null;
}

export const api = {
  health: () => request<Health>("/health"),
  settings: () => request<SettingsOut>("/settings"),
  saveSettings: (s: LocalSettings) =>
    request<SettingsOut>("/settings", { method: "PUT", ...json(s) }),
  llmModels: () => request<LlmModels>("/llm/models"),
  deleteAllData: () =>
    request<{ status: string }>("/data", { method: "DELETE", ...json({ confirm: "DELETE" }) }),

  bowlers: () => request<BowlerRecord[]>("/bowlers"),
  createBowler: (b: BowlerIn) => request<BowlerRecord>("/bowlers", { method: "POST", ...json(b) }),
  workload: (id: string) => request<WorkloadOut>(`/bowlers/${encodeURIComponent(id)}/workload`),

  sessions: (bowlerId?: string) =>
    request<SessionRecord[]>(
      bowlerId ? `/sessions?bowler_id=${encodeURIComponent(bowlerId)}` : "/sessions",
    ),
  session: (id: string) => request<SessionOut>(`/sessions/${encodeURIComponent(id)}`),
  deleteSession: (id: string) =>
    request<{ status: string }>(`/sessions/${encodeURIComponent(id)}`, { method: "DELETE" }),
  saveCalibration: (id: string, body: CalibrationIn) =>
    request<SessionOut>(`/sessions/${encodeURIComponent(id)}/calibration`, {
      method: "POST",
      ...json(body),
    }),
  coachHistory: (id: string) => request<ChatTurn[]>(`/sessions/${encodeURIComponent(id)}/coach`),
  coach: (id: string, message: string) =>
    request<CoachOut>(`/sessions/${encodeURIComponent(id)}/coach`, {
      method: "POST",
      ...json({ message }),
    }),
  people: (id: string) => request<PeopleTracks>(`/sessions/${encodeURIComponent(id)}/people`),
  chooseBowler: (id: string, trackId: number) =>
    request<SessionOut>(`/sessions/${encodeURIComponent(id)}/bowler`, {
      method: "POST",
      ...json({ track_id: trackId }),
    }),
  delivery: (id: string) => request<DeliveryOut>(`/deliveries/${encodeURIComponent(id)}`),
  setRelease: (id: string, frame: number) =>
    request<DeliveryOut>(`/deliveries/${encodeURIComponent(id)}/release`, {
      method: "POST",
      ...json({ frame }),
    }),
  annotations: (id: string) =>
    request<AnnotationsIn>(`/deliveries/${encodeURIComponent(id)}/annotations`),
  saveAnnotations: (id: string, annotations: Annotation[]) =>
    request<AnnotationsIn>(`/deliveries/${encodeURIComponent(id)}/annotations`, {
      method: "PUT",
      ...json({ annotations }),
    }),
  progress: (id: string) => request<ProgressOut>(`/bowlers/${encodeURIComponent(id)}/progress`),
  evaluation: () => request<EvaluationReport>("/evaluation"),
  drills: () => request<Record<string, Drill[]>>("/drills"),
  references: () => request<ReferencesOut>("/references"),

  reportUrl: (id: string) => `${apiBase()}/sessions/${encodeURIComponent(id)}/report.pdf`,
  clipUrl: (id: string, format: "mp4" | "gif") =>
    `${apiBase()}/deliveries/${encodeURIComponent(id)}/clip?format=${format}`,
  videoUrl: (id: string) => `${apiBase()}/sessions/${encodeURIComponent(id)}/video`,
  eventsUrl: (id: string) => `${apiBase()}/sessions/${encodeURIComponent(id)}/events`,
};

export interface UploadFields {
  file: File;
  poseFile?: File | null;
  bowlerId?: string;
  bowlingArm?: "right" | "left";
  allowLowFps?: boolean;
  /** Times slowed down; undefined = detect it, 1 = real time. */
  slowMotion?: number;
  cameraView?: "side" | "front" | "behind";
}

/** Multipart upload with progress (XHR, since fetch has no upload progress). */
export function uploadSession(
  fields: UploadFields,
  onProgress: (fraction: number) => void,
): { promise: Promise<SessionOut>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const form = new FormData();
  form.append("file", fields.file);
  if (fields.poseFile) form.append("pose_file", fields.poseFile);
  if (fields.bowlerId) form.append("bowler_id", fields.bowlerId);
  if (fields.bowlingArm) form.append("bowling_arm", fields.bowlingArm);
  if (fields.allowLowFps) form.append("allow_low_fps", "true");
  if (fields.slowMotion) form.append("slow_motion", String(fields.slowMotion));
  if (fields.cameraView) form.append("camera_view", fields.cameraView);
  const promise = new Promise<SessionOut>((resolve, reject) => {
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // fall through
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as SessionOut);
      } else {
        const detail =
          body && typeof body === "object" && "detail" in body
            ? String((body as { detail: unknown }).detail)
            : `${xhr.status} ${xhr.statusText}`;
        reject(new ApiError(xhr.status, detail));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, `Can't reach the BowlSense API at ${apiBase()}.`));
    xhr.onabort = () => reject(new ApiError(0, "Upload cancelled."));
  });
  xhr.open("POST", `${apiBase()}/sessions`);
  xhr.send(form);
  return { promise, abort: () => xhr.abort() };
}

export interface ImageFields {
  file: File;
  phase: "auto" | "bfc" | "ffc" | "release";
  view: "auto" | "side" | "front";
  bowlingArm?: "right" | "left";
}

/** Image mode: analyse one photo (synchronous on the server). */
export async function analyzeImage(fields: ImageFields): Promise<ImageAnalysis> {
  const form = new FormData();
  form.append("file", fields.file);
  form.append("phase", fields.phase);
  form.append("view", fields.view);
  if (fields.bowlingArm) form.append("bowling_arm", fields.bowlingArm);
  let res: Response;
  try {
    res = await fetch(`${apiBase()}/analyze/image`, { method: "POST", body: form });
  } catch {
    throw new ApiError(0, `Can't reach the BowlSense API at ${apiBase()}.`);
  }
  if (!res.ok) throw await errorFrom(res);
  return (await res.json()) as ImageAnalysis;
}

/** Human-readable error text for any thrown value. */
export function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
