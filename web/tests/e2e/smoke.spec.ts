import path from "node:path";
import { expect, test } from "@playwright/test";

const data = path.resolve(process.cwd(), "test-results", "e2e-stack"); // Playwright runs from web/

test("upload a synthetic delivery and see the results", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Analyse a bowling action" })).toBeVisible();
  await page.getByTestId("video-input").setInputFiles(path.join(data, "delivery.mp4"));
  await page.getByText("Advanced: import keypoints").click();
  await page.getByTestId("pose-input").setInputFiles(path.join(data, "delivery.pose.json"));
  await page.getByRole("button", { name: "Upload and analyse" }).click();

  await page.waitForURL(/\/sessions\/[0-9a-f-]{36}$/);
  await expect(page.getByTestId("technique-score")).toBeVisible({ timeout: 200_000 });
  await expect(page.getByText("Flagged: 1 of 1")).toBeVisible();
  await expect(page.getByTestId("coach-report")).toContainText("not medical advice");
  await page.screenshot({ path: "test-results/screens/session.png", fullPage: true });

  await page.getByRole("link", { name: "Details" }).first().click();
  await expect(page.getByRole("heading", { name: /Delivery 1 of 1/ })).toBeVisible();
  await expect(page.getByText("Front knee collapse").first()).toBeVisible();
  await expect(page.getByText("Bowling elbow angle")).toBeVisible();
  await expect(page.locator("video")).toHaveCount(1);
  await page.screenshot({ path: "test-results/screens/delivery.png", fullPage: true });

  await page.getByRole("link", { name: "Sessions" }).click();
  await expect(page.getByRole("link", { name: "delivery.mp4" })).toBeVisible();
  for (const name of ["Bowlers", "Compare", "Settings"]) {
    await page.getByRole("link", { name, exact: true }).click();
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  }
  expect(errors).toEqual([]);
});
