"use client";

import { useState } from "react";
import { api, errorText } from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { Button } from "./ui/Button";

/** Choose an existing bowler profile or create one inline. */
export function BowlerPicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (id: string) => void;
}) {
  const bowlers = useAsync(() => api.bowlers(), []);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [arm, setArm] = useState<"right" | "left">("right");
  const [error, setError] = useState<string | null>(null);

  async function create() {
    setError(null);
    try {
      const b = await api.createBowler({
        name: name.trim(),
        bowling_arm: arm,
        height_cm: null,
        bowling_type: null,
        dob: null,
      });
      bowlers.reload();
      onChange(b.id);
      setCreating(false);
      setName("");
    } catch (err) {
      setError(errorText(err));
    }
  }

  return (
    <div className="space-y-2 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor="bowler" className="sr-only">
          Bowler
        </label>
        <select
          id="bowler"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="min-w-52 border border-border bg-surface-2 px-3 py-2"
        >
          <option value="">No profile (one-off analysis)</option>
          {(bowlers.data ?? []).map((b) => (
            <option key={b.id} value={b.id}>
              {b.name}
              {b.bowling_arm ? ` (${b.bowling_arm}-arm)` : ""}
            </option>
          ))}
        </select>
        <Button variant="secondary" onClick={() => setCreating((c) => !c)}>
          {creating ? "Cancel" : "New bowler"}
        </Button>
      </div>
      {bowlers.error && <p className="text-bad">Couldn&apos;t load bowlers: {bowlers.error}</p>}
      {creating && (
        <div className="flex flex-wrap items-end gap-2 border border-border p-3">
          <label className="flex flex-col gap-1">
            Name
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="border border-border bg-surface-2 px-2 py-1.5"
            />
          </label>
          <label className="flex flex-col gap-1">
            Bowling arm
            <select
              value={arm}
              onChange={(e) => setArm(e.target.value as "right" | "left")}
              className="border border-border bg-surface-2 px-2 py-1.5"
            >
              <option value="right">Right</option>
              <option value="left">Left</option>
            </select>
          </label>
          <Button onClick={create} disabled={!name.trim()}>
            Save
          </Button>
          {error && <p className="w-full text-bad">{error}</p>}
        </div>
      )}
    </div>
  );
}
