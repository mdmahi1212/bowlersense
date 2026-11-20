import { describe, expect, it } from "vitest";
import { bandOf, bandStyle, cn, fmt, fmtBytes, PHASE_EVENTS } from "@/lib/utils";

describe("utils", () => {
  it("bands risk scores with the configured cut-offs", () => {
    expect(bandOf(0)).toBe("low");
    expect(bandOf(34.9)).toBe("low");
    expect(bandOf(35)).toBe("medium");
    expect(bandOf(65)).toBe("medium");
    expect(bandOf(65.1)).toBe("high");
    for (const band of ["low", "medium", "high"] as const) {
      expect(bandStyle[band].label.length).toBeGreaterThan(0); // colour always paired with text
    }
  });

  it("formats numbers and missing values", () => {
    expect(fmt(12.345, 1, "°")).toBe("12.3°");
    expect(fmt(null)).toBe("—");
    expect(fmt(undefined)).toBe("—");
    expect(fmt(Number.NaN)).toBe("—");
    expect(fmtBytes(0)).toBe("0 B");
    expect(fmtBytes(1536)).toBe("1.5 KB");
    expect(fmtBytes(5 * 1024 ** 3)).toBe("5.0 GB");
  });

  it("joins class names", () => {
    expect(cn("a", false, null, undefined, "b")).toBe("a b");
  });

  it("lists phase events in time order", () => {
    expect(PHASE_EVENTS.map(([k]) => k)).toEqual([
      "run_up_start",
      "bound",
      "bfc",
      "ffc",
      "arm_horizontal",
      "release",
      "follow_through_end",
    ]);
  });
});
