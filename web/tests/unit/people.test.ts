import { describe, expect, it } from "vitest";
import { boxesAt, busiestFrame } from "@/components/PersonPicker";
import type { PeopleTracks } from "@/lib/types";

const people: PeopleTracks = {
  width: 100,
  height: 50,
  fps: 30,
  frame_count: 5,
  chosen_track_id: 1,
  chosen_by: "auto",
  people: [
    {
      track_id: 1,
      frames: [0, 1, 2],
      boxes: [
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 3, 3],
      ],
    },
    {
      track_id: 2,
      frames: [2, 3],
      boxes: [
        [5, 5, 6, 6],
        [6, 6, 7, 7],
      ],
    },
  ],
};

describe("person picker", () => {
  it("finds each person's box at a frame", () => {
    expect([...boxesAt(people, 2).keys()]).toEqual([1, 2]);
    expect(boxesAt(people, 3).get(2)).toEqual([6, 6, 7, 7]);
    expect(boxesAt(people, 4).size).toBe(0);
  });

  it("starts on the frame with the most people", () => {
    expect(busiestFrame(people)).toBe(2);
  });
});

import { sessionVerdict, VERDICT_STYLE } from "@/components/LegalityVerdict";

describe("legality verdict", () => {
  it("says NOT LEGAL if any delivery is, LEGAL if any could be judged, else CAN'T JUDGE", () => {
    expect(sessionVerdict(["legal", "not_legal", "cannot_judge"])).toBe("not_legal");
    expect(sessionVerdict(["legal", "cannot_judge"])).toBe("legal");
    expect(sessionVerdict(["cannot_judge"])).toBe("cannot_judge");
    expect(VERDICT_STYLE.not_legal.word).toBe("NOT LEGAL");
  });
});
