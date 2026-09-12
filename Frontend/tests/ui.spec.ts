import { test, expect } from "@playwright/test";

async function mountEngine(
  page: import("@playwright/test").Page,
  permissions: string[],
  rejectBranchCreate = false,
  reservedIds = false,
) {
  await page.addInitScript(
    ({ initialPermissions, rejectBranch, useReservedIds }) => {
      const state = {
        permissions: initialPermissions,
        calls: [] as string[],
        invocations: [] as { method: string; args: unknown[] }[],
        rejectBranch,
      };
      (window as any).__argusTest = state;
      const branchId = useReservedIds ? "virtual:all" : "branch-1";
      const areaId = useReservedIds ? "id:anything" : "area-1";
      const camera = {
        id: "camera-1",
        source: "0",
        area_id: areaId,
        branch_id: branchId,
      };
      const hierarchy = {
        organization: { id: "org-1", name: "Test Organization" },
        branches: [
          {
            id: branchId,
            name: useReservedIds ? "virtual:all" : "Test Branch",
            areas: [
              {
                id: areaId,
                name: useReservedIds ? "id:anything" : "Test Area",
                branch_id: branchId,
                cameras: [camera],
              },
            ],
          },
        ],
        unassigned_cameras: [],
      };
      (window as any).argusDesktop = {
        invoke: async (method: string, args: unknown[] = []) => {
          state.calls.push(method);
          state.invocations.push({ method, args });
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
            return [
              {
                id: areaId,
                name: useReservedIds ? "id:anything" : "Test Area",
              },
            ];
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
    {
      initialPermissions: permissions,
      rejectBranch: rejectBranchCreate,
      useReservedIds: reservedIds,
    },
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

test("reserved-looking hierarchy IDs reach mutations unchanged", async ({
  page,
}) => {
  await mountEngine(page, ["view_live", "configure_cameras"], false, true);
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  const areaForm = page
    .locator("form")
    .filter({ has: page.getByLabel("New area", { exact: true }) });
  await areaForm.getByLabel("New area", { exact: true }).fill("Exact Area");
  await areaForm.locator("select").selectOption({
    label: "virtual:all",
  });
  await areaForm.getByRole("button", { name: "Add area", exact: true }).click();
  await expect
    .poll(() =>
      page.evaluate(() => {
        const invocation = (window as any).__argusTest.invocations.find(
          (item: { method: string }) => item.method === "create_area",
        );
        return invocation?.args?.[0]?.branch_id;
      }),
    )
    .toBe("virtual:all");

  await page.getByRole("button", { name: "Cameras", exact: true }).click();
  await page.getByRole("button", { name: "Add camera", exact: true }).click();
  const drawer = page.getByRole("dialog", { name: "Connect a camera" });
  await drawer.getByLabel("Camera name", { exact: true }).fill("reserved_cam");
  await drawer.getByLabel("Camera source", { exact: true }).fill("0");
  await drawer.locator("select").nth(0).selectOption({
    label: "virtual:all",
  });
  await drawer.locator("select").nth(1).selectOption({
    label: "id:anything",
  });
  await drawer.getByRole("button", { name: "Add camera", exact: true }).click();
  await expect
    .poll(() =>
      page.evaluate(() => {
        const invocation = (window as any).__argusTest.invocations.find(
          (item: { method: string }) => item.method === "add_camera",
        );
        return invocation?.args?.[0]?.area_id;
      }),
    )
    .toBe("id:anything");
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

test("streams-only wall supports filtering, focus, fullscreen, and ordered escape", async ({
  page,
}) => {
  test.slow();
  await page.getByRole("button", { name: "Open streams wall" }).click();

  const wall = page.getByRole("region", { name: "Streams-only camera wall" });
  await expect(wall).toBeVisible();
  await expect(
    page.locator(".sidebar, .topbar, .activity, .app-footer"),
  ).toHaveCount(0);
  await expect(
    wall.getByText("Deluxe Paints Nigeria", { exact: true }),
  ).toBeVisible();

  await wall
    .getByLabel("Wall branch")
    .selectOption({ label: "Ikeja Outlet (3)" });
  await wall
    .getByLabel("Wall area")
    .selectOption({ label: "External perimeter (2)" });
  await expect(wall.locator(".streams-tile")).toHaveCount(2);
  await wall.getByLabel("Search streams").fill("Main Corridor");
  await expect(wall.locator(".streams-tile")).toHaveCount(1);
  await expect(wall.getByTitle("Healthy cameras")).toHaveText("1");

  await wall.getByRole("button", { name: "Show 9 cameras" }).click();
  await expect(
    wall.getByRole("button", { name: "Show 9 cameras" }),
  ).toHaveAttribute("aria-pressed", "true");
  await wall.getByRole("button", { name: "Focus Main Corridor" }).click();
  await expect(wall.locator(".streams-stage")).toHaveClass(/is-focused/);

  await page.evaluate(() => {
    let fullscreenElement: Element | null = null;
    Object.defineProperty(document, "fullscreenElement", {
      configurable: true,
      get: () => fullscreenElement,
    });
    Object.defineProperty(Element.prototype, "requestFullscreen", {
      configurable: true,
      value: async function () {
        fullscreenElement = this;
        document.dispatchEvent(new Event("fullscreenchange"));
      },
    });
    Object.defineProperty(document, "exitFullscreen", {
      configurable: true,
      value: async () => {
        fullscreenElement = null;
        document.dispatchEvent(new Event("fullscreenchange"));
      },
    });
  });
  await wall.getByRole("button", { name: "Enter fullscreen" }).click();
  await expect(wall).toHaveAttribute("data-fullscreen", "true");

  await page.keyboard.press("Escape");
  await expect(wall.locator(".streams-stage")).not.toHaveClass(/is-focused/);
  await expect(wall).toHaveAttribute("data-fullscreen", "true");
  await page.keyboard.press("Escape");
  await expect(wall).toHaveAttribute("data-fullscreen", "false");
  await expect(wall).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(wall).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "Every camera. One clear picture." }),
  ).toBeVisible();
});

test("streams-only focus reconciles after search, branch, and area exclusion", async ({
  page,
}) => {
  await page.getByRole("button", { name: "Open streams wall" }).click();
  const wall = page.getByRole("region", { name: "Streams-only camera wall" });
  const stage = wall.locator(".streams-stage");
  const focusLoadingBay = () =>
    wall.getByRole("button", { name: "Focus Loading Bay" }).click();

  await focusLoadingBay();
  await wall.getByLabel("Search streams").fill("Forecourt ATM");
  await expect(stage).not.toHaveClass(/is-focused/);
  await expect(wall.getByText("Loading Bay", { exact: true })).toHaveCount(0);

  await wall.getByLabel("Search streams").fill("");
  await focusLoadingBay();
  await wall
    .getByLabel("Wall branch")
    .selectOption({ label: "Ikeja Outlet (3)" });
  await expect(stage).not.toHaveClass(/is-focused/);
  await expect(wall.getByText("Loading Bay", { exact: true })).toHaveCount(0);

  await wall
    .getByLabel("Wall branch")
    .selectOption({ label: "All branches (4)" });
  await focusLoadingBay();
  await wall
    .getByLabel("Wall area")
    .selectOption({ label: "Retail floor (1)" });
  await expect(stage).not.toHaveClass(/is-focused/);
  await expect(wall.getByText("Loading Bay", { exact: true })).toHaveCount(0);
});

test("streams-only toolbar auto-hides without overlap and remains visible while focused", async ({
  page,
}) => {
  test.slow();
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole("button", { name: "Open streams wall" }).click();
  const wall = page.getByRole("region", { name: "Streams-only camera wall" });
  const toolbar = wall.getByRole("toolbar", { name: "Camera wall controls" });
  await expect(toolbar).toBeVisible();
  await page.waitForTimeout(3200);
  await expect(toolbar).toHaveAttribute("data-hidden", "true");

  await page.mouse.move(20, 20);
  await expect(toolbar).toHaveAttribute("data-hidden", "false");
  await wall.getByLabel("Search streams").focus();
  await page.waitForTimeout(3200);
  await expect(toolbar).toHaveAttribute("data-hidden", "false");

  await expect
    .poll(() =>
      wall.locator(".streams-tile .camera-media").evaluateAll((nodes) =>
        nodes.every((node) => {
          const rect = node.getBoundingClientRect();
          return rect.width > 100 && rect.height > 60;
        }),
      ),
    )
    .toBe(true);
  await expect
    .poll(() =>
      wall.locator(".streams-tile video").evaluateAll((nodes) =>
        nodes.every((node) => {
          const video = node as HTMLVideoElement;
          return (
            video.readyState >= 2 &&
            video.videoWidth > 0 &&
            video.videoHeight > 0
          );
        }),
      ),
    )
    .toBe(true);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  await page.screenshot({
    path: "test-results/streams-wall-desktop.png",
    fullPage: true,
  });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.mouse.move(10, 120);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);
  const boxes = await wall
    .locator(".streams-tile .camera-media")
    .evaluateAll((nodes) =>
      nodes.map((node) => {
        const rect = node.getBoundingClientRect();
        return { width: rect.width, height: rect.height };
      }),
    );
  expect(boxes.every(({ width, height }) => width > 100 && height > 60)).toBe(
    true,
  );
  await page.screenshot({
    path: "test-results/streams-wall-mobile.png",
    fullPage: true,
  });
});

test("streams-only density 16 keeps mobile focus actions and stable tiles", async ({
  page,
}) => {
  await page.evaluate(async () => {
    const { initialDemo } = await import("/src/lib/demo.ts");
    const state = initialDemo();
    const source = state.cameras[0];
    state.cameras = Array.from({ length: 16 }, (_, index) => ({
      ...source,
      id: `Camera ${String(index + 1).padStart(2, "0")}`,
    }));
    localStorage.setItem("argus.desktop.demo.v1", JSON.stringify(state));
  });
  await page.reload();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open streams wall" }).click();
  const wall = page.getByRole("region", { name: "Streams-only camera wall" });
  await wall.getByRole("button", { name: "Show 16 cameras" }).click();

  const tiles = wall.locator(".streams-tile");
  const focusActions = wall.getByRole("button", { name: /^Focus Camera/ });
  await expect(tiles).toHaveCount(16);
  await expect(focusActions).toHaveCount(16);
  await expect(focusActions.first()).toBeVisible();
  await expect(tiles.locator(".streams-caption strong")).toHaveCount(16);

  const boxes = await tiles.evaluateAll((nodes) =>
    nodes.map((node) => {
      const rect = node.getBoundingClientRect();
      return { width: rect.width, height: rect.height };
    }),
  );
  expect(boxes.every(({ width, height }) => width > 80 && height > 110)).toBe(
    true,
  );
  expect(Math.max(...boxes.map(({ width }) => width))).toBeCloseTo(
    Math.min(...boxes.map(({ width }) => width)),
    0,
  );
  expect(Math.max(...boxes.map(({ height }) => height))).toBeCloseTo(
    Math.min(...boxes.map(({ height }) => height)),
    0,
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/streams-wall-mobile-density-16.png",
    fullPage: true,
  });

  await focusActions.first().click();
  await expect(wall.locator(".streams-stage")).toHaveClass(/is-focused/);
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
