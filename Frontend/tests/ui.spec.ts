import { test, expect } from "@playwright/test";

async function mountEngine(
  page: import("@playwright/test").Page,
  permissions: string[],
  rejectBranchCreate = false,
) {
  await page.addInitScript(
    ({ initialPermissions, rejectBranch }) => {
      const state = {
        permissions: initialPermissions,
        calls: [] as string[],
        rejectBranch,
      };
      (window as any).__argusTest = state;
      const camera = {
        id: "camera-1",
        source: "0",
        area_id: "area-1",
        branch_id: "branch-1",
      };
      const hierarchy = {
        organization: { id: "org-1", name: "Test Organization" },
        branches: [
          {
            id: "branch-1",
            name: "Test Branch",
            areas: [
              {
                id: "area-1",
                name: "Test Area",
                branch_id: "branch-1",
                cameras: [camera],
              },
            ],
          },
        ],
        unassigned_cameras: [],
      };
      (window as any).argusDesktop = {
        invoke: async (method: string) => {
          state.calls.push(method);
          if (method === "auth_state")
            return {
              configured: true,
              signed_in: true,
              username: "mounted-user",
              role: "custom",
              permissions: [...state.permissions],
            };
          if (method === "list_cameras") return [camera];
          if (method === "list_events") return [];
          if (method === "list_areas")
            return [{ id: "area-1", name: "Test Area" }];
          if (method === "hierarchy") return hierarchy;
          if (method === "get_site")
            return { name: "Test Site", notify: "console" };
          if (method === "monitoring_status") return { running: false };
          if (method === "english_rules_status") return { available: false };
          if (method === "feed_sources") return { active: "", sources: [] };
          if (method === "gate_status") return { ready: false };
          if (method === "use_case_templates") return {};
          if (method === "camera_stream")
            return { kind: "mjpeg", url: "http://127.0.0.1:9/frame" };
          if (method === "create_branch" && state.rejectBranch)
            throw Object.assign(
              new Error("Forbidden (requires configure_cameras)"),
              {
                status: 403,
                code: "forbidden",
                permission: "configure_cameras",
                detail: { permission: "configure_cameras" },
              },
            );
          return { ok: true };
        },
        subscribe: () => () => {},
        environment: async () => ({}),
      };
    },
    { initialPermissions: permissions, rejectBranch: rejectBranchCreate },
  );
  await page.reload();
  await page.getByRole("button", { name: "Local engine", exact: true }).click();
  await expect(
    page.getByText("Backend connected", { exact: true }),
  ).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Every camera. One clear picture." }),
  ).toBeVisible();
});

test("operator sees hierarchy without camera setup mutation entry points", async ({
  page,
}) => {
  await mountEngine(page, ["view_live"]);
  await page.getByRole("button", { name: "Cameras", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Add camera", exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await expect(
    page.getByText("Test Organization", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Site setup", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Add branch", exact: true }),
  ).toHaveCount(0);
});

test("permission revocation closes camera onboarding and prevents invocation", async ({
  page,
}) => {
  await mountEngine(page, ["view_live", "configure_cameras"]);
  await page.getByRole("button", { name: "Cameras", exact: true }).click();
  await page.getByRole("button", { name: "Add camera", exact: true }).click();
  await expect(
    page.getByRole("dialog", { name: "Connect a camera" }),
  ).toBeVisible();
  await page.evaluate(() => {
    (window as any).__argusTest.permissions = ["view_live"];
  });
  await expect(
    page.getByRole("dialog", { name: "Connect a camera" }),
  ).toHaveCount(0, { timeout: 12_000 });
  expect(
    await page.evaluate(
      () =>
        (window as any).__argusTest.calls.filter(
          (method: string) => method === "add_camera",
        ).length,
    ),
  ).toBe(0);
});

test("structured branch 403 leaves mounted hierarchy unchanged", async ({
  page,
}) => {
  await mountEngine(page, ["view_live", "configure_cameras"], true);
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByLabel("New branch", { exact: true }).fill("Blocked Branch");
  await page.getByRole("button", { name: "Add branch", exact: true }).click();
  await expect(
    page.getByText("Forbidden (requires configure_cameras)", { exact: true }),
  ).toBeVisible();
  const branchName = page.locator(".location-branch-head input");
  await expect(branchName).toHaveCount(1);
  await expect(branchName).toHaveValue("Test Branch");
  await expect(branchName).not.toHaveValue("Blocked Branch");
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
test("existing backend polygon zones render without monitoring", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.evaluate(async () => {
    const { initialDemo } = await import("/src/lib/demo.ts");
    const state = initialDemo();
    state.zones["Loading Bay"] = [
      {
        name: "Existing zone",
        polygon: [
          [0, 0],
          [400, 0],
          [400, 300],
          [0, 300],
        ],
        dwell_alert_seconds: 5,
      },
    ];
    localStorage.setItem("argus.desktop.demo.v1", JSON.stringify(state));
  });
  await page.reload();
  await page
    .getByRole("button", { name: "Configure Loading Bay", exact: true })
    .click();
  await page.getByRole("tab", { name: "Zones", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Remove Existing zone" }),
  ).toBeVisible();
  await expect(page.locator(".saved-zone")).toHaveAttribute(
    "points",
    "0,0 400,0 400,300 0,300",
  );
  expect(errors).toEqual([]);
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
      ][0].polygon,
  );
  expect(points).toHaveLength(4);
  expect(points[2][0]).toBeGreaterThan(points[0][0]);
});
test("zone workspace fits the entire image and protects unsaved drawings", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page
    .getByRole("button", { name: "Configure Loading Bay", exact: true })
    .click();
  await page.getByRole("tab", { name: "Zones", exact: true }).click();
  const canvas = page.getByRole("img", { name: "Zone drawing canvas" });
  await expect(canvas).toBeVisible();
  await expect
    .poll(async () => (await page.getByRole("dialog").boundingBox())!.width)
    .toBeGreaterThan(1400);
  for (const size of [
    { width: 1440, height: 900 },
    { width: 1024, height: 768 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(size);
    await expect
      .poll(async () => {
        const box = (await canvas.boundingBox())!;
        const ratio = await page
          .locator(".zone-canvas > img")
          .evaluate(
            (img: HTMLImageElement) => img.naturalWidth / img.naturalHeight,
          );
        return (
          box.y >= 0 &&
          box.y + box.height <= size.height &&
          box.x >= 0 &&
          box.x + box.width <= size.width &&
          Math.abs(box.width / box.height - ratio) < 0.01
        );
      })
      .toBe(true);
  }
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole("button", { name: "Draw rectangle" }).click();
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + 10, box.y + 10);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width + 10, box.y + box.height + 10);
  await page.mouse.up();
  page.once("dialog", (dialog) => dialog.dismiss());
  await page.keyboard.press("Escape");
  await expect(canvas).toBeVisible();
  await page.getByLabel("Zone name", { exact: true }).fill("Bottom edge");
  await page.getByRole("button", { name: "Save zone", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Remove Bottom edge" }),
  ).toBeVisible();
  const height = await page
    .locator(".zone-canvas > img")
    .evaluate((img: HTMLImageElement) => img.naturalHeight);
  const polygon = await page.evaluate(
    () =>
      JSON.parse(localStorage.getItem("argus.desktop.demo.v1")!).zones[
        "Loading Bay"
      ][0].polygon,
  );
  expect(polygon[2][1]).toBe(height - 1);
  await page.screenshot({ path: "test-results/zones-fullscreen.png" });
  await page.getByRole("button", { name: "Close details" }).click();
  await page
    .getByRole("button", { name: "Configure Forecourt ATM", exact: true })
    .click();
  await page.getByRole("tab", { name: "Zones", exact: true }).click();
  await expect(canvas).toBeVisible();
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(viewport);
    await expect
      .poll(async () => {
        const bounds = (await canvas.boundingBox())!;
        return (
          bounds.y + bounds.height <= viewport.height &&
          Math.abs(bounds.width / bounds.height - 16 / 9) < 0.01
        );
      })
      .toBe(true);
    await page.screenshot({
      path: `test-results/zones-landscape-${viewport.width}.png`,
    });
  }
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
