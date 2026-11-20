"use client";

import { useRef, useState, type DragEvent } from "react";
import { BowlingLoop } from "@/components/art/BowlingLoop";
import { cn } from "@/lib/utils";

const ACCEPT =
  ".mp4,.mov,.avi,.webm,.jpg,.jpeg,.png,video/mp4,video/quicktime,video/x-msvideo,video/webm,image/jpeg,image/png";
const EXTENSIONS = [".mp4", ".mov", ".avi", ".webm", ".jpg", ".jpeg", ".png"];

/** True for the photo types that go to image mode. */
export const isImage = (f: File): boolean => /\.(jpe?g|png)$/i.test(f.name);

/** Drag-and-drop (or click) video picker; rejects other file types before uploading. */
export function UploadDropzone({
  file,
  onFile,
  onFiles,
  disabled,
}: {
  file: File | null;
  onFile: (file: File | null, error?: string) => void;
  /** Several videos dropped at once (batch upload). */
  onFiles?: (files: File[]) => void;
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

  function pickMany(list: FileList | null) {
    const all = Array.from(list ?? []);
    const videos = all.filter(
      (f) => !isImage(f) && EXTENSIONS.some((e) => f.name.toLowerCase().endsWith(e)),
    );
    if (all.length > 1 && onFiles) {
      if (videos.length < all.length)
        onFile(null, "Batch upload takes videos only (MP4, MOV, AVI).");
      if (videos.length) onFiles(videos);
    } else pick(all[0]);
  }

  function drop(e: DragEvent) {
    e.preventDefault();
    setOver(false);
    if (!disabled) pickMany(e.dataTransfer.files);
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
        "group relative grid cursor-pointer items-center gap-8 overflow-hidden bg-plate p-8 text-[#f2eee5] transition-[outline] sm:grid-cols-[260px_minmax(0,1fr)]",
        over
          ? "outline outline-2 outline-offset-4 outline-accent"
          : "hover:outline hover:outline-1 hover:outline-offset-4 hover:outline-foreground/40",
        disabled && "cursor-not-allowed opacity-60",
      )}
    >
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        className="hidden"
        data-testid="video-input"
        multiple={Boolean(onFiles)}
        onChange={(e) => pickMany(e.target.files)}
      />
      <BowlingLoop className="mx-auto h-52 w-auto text-[#f2eee5] sm:h-60" />
      <div>
        {file ? (
          <>
            <p className="kicker !text-[#f2eee5]/60">Ready</p>
            <p className="mt-2 break-all font-display text-2xl">{file.name}</p>
            <p className="mt-2 text-sm text-[#f2eee5]/60">
              {(file.size / 1024 / 1024).toFixed(1)} MB · click to choose another
            </p>
          </>
        ) : (
          <>
            <p className="kicker !text-[#f2eee5]/60">Step one</p>
            <p className="mt-2 font-display text-3xl leading-tight">
              Drop a bowling video here, or click to choose one.
            </p>
            <p className="mt-3 text-sm text-[#f2eee5]/60">
              MP4, MOV or AVI up to 500 MB, or several at once to queue them. A JPG or PNG gives a
              single-frame posture check.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
