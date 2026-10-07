import { test, expect } from "@playwright/test";

test("Operations detectors and shareable diagnostics are accessible", async ({ page }) => {
  await page.goto("/tests/fixtures/support.html");
  await expect(page.getByRole("heading", { name: "Operations", exact: true })).toBeVisible();
  await page.getByRole("switch", { name: "Normal movement", exact: true }).click();
  await expect.poll(() => page.evaluate(() => (window as any).supportCalls)).toContainEqual({
    method: "set_camera_rules", args: ["Bay", { normal_movement: true }],
  });
  await expect(page.getByRole("switch", { name: "Multiple people moving", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Export diagnostics" }).click();
  await expect(page.getByLabel("Diagnostics ZIP on the engine computer")).toHaveValue("C:\\Argus\\argus-diagnostics-test.zip");
  await page.screenshot({ path: "test-results/support-desktop.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator(".settings-section").screenshot({ path: "test-results/support-mobile.png" });
  await page.locator("section").filter({ has: page.getByRole("heading", { name: "Operations", exact: true }) }).screenshot({ path: "test-results/operations-mobile.png" });
});
