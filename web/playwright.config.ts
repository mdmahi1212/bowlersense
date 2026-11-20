import { defineConfig } from "@playwright/test";

// Smoke test against a throwaway local stack (tests/e2e/stack.mjs) and the production build
// (`npm run build` first). Uses the installed Microsoft Edge on Windows, otherwise Playwright's
// Chromium (`npx playwright install chromium` once), so no browser download is needed on Windows.
const apiPort = process.env.E2E_API_PORT ?? "8765";
const webPort = process.env.E2E_WEB_PORT ?? "3765";
const channel = process.env.PW_CHANNEL ?? (process.platform === "win32" ? "msedge" : undefined);

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 240_000,
  expect: { timeout: 15_000 },
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${webPort}`, channel, trace: "retain-on-failure" },
  webServer: [
    {
      command: "node tests/e2e/stack.mjs",
      url: `http://127.0.0.1:${apiPort}/health`,
      timeout: 180_000,
      reuseExistingServer: false,
      env: { E2E_API_PORT: apiPort, E2E_WEB_PORT: webPort },
    },
    {
      command: `npx next start --hostname 127.0.0.1 -p ${webPort}`,
      url: `http://127.0.0.1:${webPort}`,
      timeout: 120_000,
      reuseExistingServer: false,
      env: { BOWLSENSE_API_URL: `http://127.0.0.1:${apiPort}`, NEXT_TELEMETRY_DISABLED: "1" },
    },
  ],
});
