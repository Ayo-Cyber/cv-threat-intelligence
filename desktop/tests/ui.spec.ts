import { test, expect } from "@playwright/test";
test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Every camera. One clear picture." }),
  ).toBeVisible();
});
test("overview, real media and responsive layout", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.getByRole("button", { name: "Play footage", exact: true }).click();
  await expect(page.locator("video")).toHaveCount(4);
  await expect
    .poll(() =>
      page
        .locator("video")
        .evaluateAll((nodes) =>
          nodes.every(
            (n) =>
              (n as HTMLVideoElement).readyState >= 2 &&
              (n as HTMLVideoElement).videoWidth > 0,
          ),
        ),
    )
    .toBe(true);
  await page.screenshot({
    path: "test-results/overview-desktop.png",
    fullPage: true,
  });
  for (const width of [1440, 1024, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await expect
      .poll(() =>
        page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      )
      .toBe(true);
  }
  await page.screenshot({
    path: "test-results/overview-mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});
test("scene edits persist and remap cannot fake inference", async ({
  page,
}) => {
  await page
    .getByRole("button", { name: "Configure Loading Bay", exact: true })
    .click();
  await page
    .getByLabel("Scene description")
    .fill("Warehouse with a loading route.");
  await page.waitForTimeout(4500);
  await expect(page.getByLabel("Scene description")).toHaveValue(
    "Warehouse with a loading route.",
  );
  await page.getByRole("button", { name: "Save & approve" }).click();
  await expect(page.getByText("Human reviewed", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Remap", exact: true }).click();
  await expect(
    page.getByText(/AI remapping requires the local engine/),
  ).toBeVisible();
  await page.screenshot({
    path: "test-results/scene-review.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Close details" }).click();
  await page.reload();
  await page
    .getByRole("button", { name: "Configure Loading Bay", exact: true })
    .click();
  await expect(page.getByLabel("Scene description")).toHaveValue(
    "Warehouse with a loading route.",
  );
});
test("detector switches and English rules survive reopening", async ({
  page,
}) => {
  await page
    .getByRole("button", { name: "Configure Loading Bay", exact: true })
    .click();
  await page.getByRole("tab", { name: "Detectors", exact: true }).click();
  await page.getByRole("switch", { name: "Fire & smoke", exact: true }).click();
  await expect(
    page.getByRole("switch", { name: "Fire & smoke", exact: true }),
  ).toHaveAttribute("aria-checked", "true");
  await page.getByRole("tab", { name: "English rules", exact: true }).click();
  await page
    .getByLabel("Watch condition")
    .fill("A person is carrying a ladder.");
  await page.getByRole("button", { name: "Add watch condition" }).click();
  await expect(
    page.getByText("A person is carrying a ladder.", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close details" }).click();
  await page.reload();
  await page
    .getByRole("button", { name: "Configure Loading Bay", exact: true })
    .click();
  await page.getByRole("tab", { name: "English rules", exact: true }).click();
  await expect(
    page.getByText("A person is carrying a ladder.", { exact: true }),
  ).toBeVisible();
});
test("rectangle zoning saves original image points", async ({ page }) => {
  await page
    .getByRole("button", { name: "Configure Loading Bay", exact: true })
    .click();
  await page.getByRole("tab", { name: "Zones", exact: true }).click();
  await page.getByRole("button", { name: "Draw rectangle" }).click();
  const canvas = page.getByRole("img", { name: "Zone drawing canvas" });
  await expect(canvas).toBeVisible();
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.15, box.y + box.height * 0.15);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.75, box.y + box.height * 0.75, {
    steps: 8,
  });
  await page.mouse.up();
  await page.getByLabel("Zone name").fill("Reception");
  await page.getByRole("button", { name: "Save zone", exact: true }).click();
  await expect(page.getByText(/Loitering after 5s/)).toBeVisible();
  await page.screenshot({
    path: "test-results/zone-editor.png",
    fullPage: true,
  });
  const points = await page.evaluate(
    () =>
      JSON.parse(localStorage.getItem("argus.desktop.demo.v1")!).zones[
        "Loading Bay"
      ][0].points,
  );
  expect(points).toHaveLength(4);
  expect(points[2][0]).toBeGreaterThan(points[0][0]);
});
test("incident review persists a false alarm decision", async ({ page }) => {
  await page.getByRole("button", { name: /SAMPLE ATM interference/ }).click();
  await page.getByLabel("Operator note").fill("No conclusive intrusion.");
  await page.getByRole("button", { name: "False alarm", exact: true }).click();
  await expect(page.getByText("Review saved.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Close details" }).click();
  await page
    .getByRole("button", { name: "All incidents", exact: true })
    .click();
  await page.getByRole("button", { name: "Resolved", exact: true }).click();
  await expect(
    page.getByRole("button", { name: /ATM interference/ }),
  ).toBeVisible();
});
test("browser cannot claim to connect to local engine", async ({ page }) => {
  await page.getByRole("button", { name: "Local engine", exact: true }).click();
  await expect(page.getByText(/Launch the Electron app/)).toBeVisible();
  await expect(
    page.getByText("Backend connected", { exact: true }),
  ).toHaveCount(0);
});

test("theme, onboarding and unsaved scene protection", async ({ page }) => {
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.screenshot({
    path: "test-results/overview-dark.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Configure Loading Bay", exact: true })
    .click();
  await page.getByLabel("Scene description").fill("Unfinished scene draft");
  page.once("dialog", (dialog) => dialog.dismiss());
  await page.getByRole("button", { name: "Close details" }).click();
  await expect(page.getByLabel("Scene description")).toHaveValue(
    "Unfinished scene draft",
  );
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Close details" }).click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Site setup", exact: true }).click();
  for (let i = 0; i < 5; i++)
    await page.getByRole("button", { name: "Continue", exact: true }).click();
  await page.getByRole("button", { name: "Finish setup", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Every camera. One clear picture." }),
  ).toBeVisible();
});
