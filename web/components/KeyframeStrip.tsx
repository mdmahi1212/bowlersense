/* eslint-disable @next/next/no-img-element -- local API images, no optimisation server */
import { fileUrl } from "@/lib/api";
import type { PhaseEvents } from "@/lib/types";
import { PHASE_EVENTS } from "@/lib/utils";

/** BFC / FFC / arm-horizontal / release / follow-through stills; click to seek the player. */
export function KeyframeStrip({
  keyframes,
  phases,
  onJump,
}: {
  keyframes: Record<string, string>;
  phases: PhaseEvents;
  onJump: (frame: number) => void;
}) {
  const items = PHASE_EVENTS.filter(([key]) => keyframes[key]);
  if (items.length === 0) return <p className="text-sm text-muted">No keyframes.</p>;
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
      {items.map(([key, label]) => (
        <button
          key={key}
          type="button"
          onClick={() => onJump(phases[key])}
          className="overflow-hidden rounded-md border border-border text-left hover:border-accent"
        >
          <img
            src={fileUrl(keyframes[key])}
            alt={`${label} keyframe`}
            className="aspect-video w-full bg-black object-contain"
          />
          <span className="block px-2 py-1 text-xs">{label}</span>
        </button>
      ))}
    </div>
  );
}
