import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, apiBase, errorText, fileUrl } from "@/lib/api";

const API_URL = apiBase();

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("talks only to the local API", () => {
    expect(API_URL).toMatch(/^http:\/\/(127\.0\.0\.1|localhost):\d+$/);
  });

  it("reads the API address from the page when the server sets it", () => {
    vi.stubGlobal("document", {
      querySelector: () => ({ content: "http://127.0.0.1:8123/" }),
    });
    expect(apiBase()).toBe("http://127.0.0.1:8123");
  });

  it("builds safe file URLs", () => {
    expect(fileUrl("outputs/abc/key frames/01 bfc.png")).toBe(
      `${API_URL}/files/outputs/abc/key%20frames/01%20bfc.png`,
    );
    expect(api.reportUrl("a/b")).toBe(`${API_URL}/sessions/a%2Fb/report.pdf`);
  });

  it("turns API errors into readable messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () => new Response(JSON.stringify({ detail: "Session not found" }), { status: 404 }),
      ),
    );
    await expect(api.session("x")).rejects.toMatchObject({
      status: 404,
      message: "Session not found",
    });
  });

  it("explains when the API is not running", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("fetch failed");
      }),
    );
    const err = await api.health().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(errorText(err)).toContain("python run.py");
  });

  it("sends JSON bodies", async () => {
    const fetchMock = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>(
      async () => new Response(JSON.stringify({ status: "deleted" })),
    );
    vi.stubGlobal("fetch", fetchMock);
    await api.deleteAllData();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_URL}/data`);
    expect(init?.method).toBe("DELETE");
    expect(JSON.parse(String(init?.body))).toEqual({ confirm: "DELETE" });
  });
});
