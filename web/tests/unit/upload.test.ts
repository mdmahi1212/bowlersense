import { describe, expect, it } from "vitest";
import { isImage } from "@/components/UploadDropzone";

const file = (name: string) => ({ name }) as File;

describe("upload routing", () => {
  it("sends photos to image mode and videos to the pipeline", () => {
    expect(isImage(file("bowler.JPG"))).toBe(true);
    expect(isImage(file("release.jpeg"))).toBe(true);
    expect(isImage(file("ffc.png"))).toBe(true);
    expect(isImage(file("nets.mp4"))).toBe(false);
    expect(isImage(file("png.mov"))).toBe(false);
  });
});
