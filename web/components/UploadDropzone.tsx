"use client";

import { useRef, useState, type DragEvent } from "react";
import { cn } from "@/lib/utils";

const ACCEPT =
  ".mp4,.mov,.avi,.jpg,.jpeg,.png,video/mp4,video/quicktime,video/x-msvideo,image/jpeg,image/png";
const EXTENSIONS = [".mp4", ".mov", ".avi", ".jpg", ".jpeg", ".png"];

/** True for the photo types that go to image mode. */
export const isImage = (f: File): boolean => /\.(jpe?g|png)$/i.test(f.name);

/** Drag-and-drop (or click) video picker; rejects other file types before uploading. */
export function UploadDropzone({
  file,
  onFile,
  disabled,
}: {
  file: File | null;
  onFile: (file: File | null, error?: string) => void;
  disabled?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  function pick(f: File | undefined) {
    if (!f) return;
    const ok = EXTENSIONS.some((e) => f.name.toLowerCase().endsWith(e));
    onFile(
      ok ? f : null,
      ok ? undefined : `${f.name} isn't supported. Use MP4, MOV, AVI, JPG or PNG.`,
    );
  }

  function drop(e: DragEvent) {
    e.preventDefault();
    setOver(false);
    if (!disabled) pick(e.dataTransfer.files[0]);
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Choose a bowling video or photo to upload"
      aria-disabled={disabled}
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !disabled) {
          e.preventDefault();
          input.current?.click();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
      className={cn(
        "flex min-h-44 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition-colors",
        over ? "border-accent bg-accent/10" : "border-border bg-surface hover:border-accent",
        disabled && "cursor-not-allowed opacity-60",
      )}
    >
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        className="hidden"
        data-testid="video-input"
        onChange={(e) => pick(e.target.files?.[0])}
      />
      {file ? (
        <>
          <p className="font-semibold">{file.name}</p>
          <p className="text-sm text-muted">
            {(file.size / 1024 / 1024).toFixed(1)} MB · click to change
          </p>
        </>
      ) : (
        <>
          <p className="font-semibold">Drop a bowling video (or a photo) here</p>
          <p className="text-sm text-muted">
            or click to choose · MP4, MOV or AVI up to 500 MB · JPG or PNG for a posture check
          </p>
        </>
      )}
    </div>
  );
}
