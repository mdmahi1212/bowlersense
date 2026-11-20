"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "./ui/Button";

/** Record a delivery with this computer's camera (nothing leaves the machine). The browser
 *  records variable-rate video; the server converts it to constant-rate MP4 using the frame
 *  rate the camera reports. */
export function WebcamRecorder({ onDone }: { onDone: (file: File, fps: number) => void }) {
  const preview = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [rec, setRec] = useState<MediaRecorder | null>(null);
  const [count, setCount] = useState<number | null>(null);
  const [seconds, setSeconds] = useState(0);
  const [fps, setFps] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream]);
  useEffect(() => {
    if (!rec) return;
    const t0 = performance.now();
    const id = setInterval(() => setSeconds((performance.now() - t0) / 1000), 200);
    return () => clearInterval(id);
  }, [rec]);

  async function open() {
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { frameRate: { ideal: 120 }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      setStream(s);
      setFps(s.getVideoTracks()[0]?.getSettings().frameRate ?? null);
      if (preview.current) {
        preview.current.srcObject = s;
        void preview.current.play();
      }
    } catch (err) {
      setError(err instanceof Error ? `Camera unavailable: ${err.message}` : "Camera unavailable.");
    }
  }

  function start() {
    if (!stream) return;
    let n = 3;
    setCount(n);
    const id = setInterval(() => {
      n -= 1;
      if (n > 0) return setCount(n);
      clearInterval(id);
      setCount(null);
      const type = ["video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm"].find((t) =>
        MediaRecorder.isTypeSupported(t),
      );
      const r = new MediaRecorder(
        stream,
        type ? { mimeType: type, videoBitsPerSecond: 12_000_000 } : undefined,
      );
      const chunks: Blob[] = [];
      r.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      r.onstop = () => {
        const mime = r.mimeType || "video/webm";
        const ext = mime.includes("mp4") ? "mp4" : "webm";
        const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
        onDone(new File(chunks, `recording-${stamp}.${ext}`, { type: mime }), fps ?? 30);
        stream.getTracks().forEach((t) => t.stop());
        setStream(null);
        setRec(null);
      };
      r.start(250);
      setSeconds(0);
      setRec(r);
    }, 1000);
  }

  if (!stream)
    return (
      <div className="flex flex-wrap items-center gap-4">
        <Button variant="secondary" onClick={open}>
          Record with this computer&apos;s camera
        </Button>
        <span className="text-xs text-muted">
          Webcams are usually 30–60 fps: fine for practice, less precise at release.
        </span>
        {error && <p className="w-full text-sm text-bad">{error}</p>}
      </div>
    );
  return (
    <div className="space-y-3">
      <div className="relative bg-plate">
        <video
          ref={preview}
          muted
          playsInline
          className="block max-h-[50vh] w-full object-contain"
        />
        {count !== null && (
          <span className="absolute inset-0 grid place-items-center font-display text-8xl text-white">
            {count}
          </span>
        )}
        {rec && (
          <span className="tabular absolute left-3 top-3 flex items-center gap-2 bg-black/70 px-2 py-1 text-sm text-white">
            <span className="h-2 w-2 animate-pulse rounded-full bg-bad" /> REC {seconds.toFixed(1)}{" "}
            s
          </span>
        )}
        <span className="tabular absolute right-3 top-3 bg-black/70 px-2 py-1 text-xs text-white">
          {fps ? `${fps.toFixed(0)} fps` : "fps unknown"}
        </span>
      </div>
      <div className="flex gap-3">
        {rec ? (
          <Button onClick={() => rec.stop()}>Stop and use this clip</Button>
        ) : (
          <Button onClick={start} disabled={count !== null}>
            Start recording (3 s countdown)
          </Button>
        )}
        <Button
          variant="ghost"
          onClick={() => {
            stream.getTracks().forEach((t) => t.stop());
            setStream(null);
            setRec(null);
          }}
        >
          Cancel
        </Button>
      </div>
    </div>
  );
}
