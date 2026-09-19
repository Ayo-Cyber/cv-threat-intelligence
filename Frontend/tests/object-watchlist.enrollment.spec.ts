import { expect, test } from "@playwright/test";
import path from "node:path";

const pngPath = path.join(process.cwd(), "public", "argus-mark.png");

type Target = {
  id: string;
  label: string;
  category: string;
  aliases: string[];
  review_state: string;
  min_similarity: number;
  allowed_zone_ids: string[];
  examples: unknown[];
  negative_examples: unknown[];
  can_activate: boolean;
  reasons: string[];
};

async function mountObjectWatchlist(
  page: import("@playwright/test").Page,
  permissions: string[],
  options: {
    failFirstExample?: boolean;
    initialTargets?: Target[];
    runtime?: Record<string, unknown>;
    jobStatuses?: Record<string, unknown>[];
  } = {},
) {
  await page.addInitScript(
    ({
      initialPermissions,
      failFirstExample,
      seededTargets,
      seededRuntime,
      jobStatuses,
    }) => {
      const targets: Target[] = [...seededTargets];
      const state = {
        invocations: [] as { method: string; args: unknown[] }[],
        failFirstExample,
        jobStatuses: [...jobStatuses] as Record<string, unknown>[],
      };
      (window as any).__argusObjectTest = state;
      const hierarchy = {
        organization: { id: "org-1", name: "Test Organization" },
        branches: [
          {
            id: "branch-1",
            name: "Main branch",
            areas: [
              {
                id: "area-1",
                name: "Warehouse area",
                branch_id: "branch-1",
                cameras: [
                  {
                    id: "camera-001",
                    source: "0",
                    area_id: "area-1",
                    branch_id: "branch-1",
                  },
                ],
              },
            ],
          },
        ],
        unassigned_cameras: [],
      };
      const cameras = hierarchy.branches[0].areas[0].cameras;
      (window as any).argusDesktop = {
        invoke: async (method: string, args: unknown[] = []) => {
          state.invocations.push({ method, args });
          if (method === "auth_state")
            return {
              configured: true,
              signed_in: true,
              username: "object-tester",
              role: initialPermissions.includes("configure_cameras")
                ? "owner"
                : "operator",
              permissions: initialPermissions,
            };
          if (method === "list_cameras") return cameras;
          if (method === "list_events") return [];
          if (method === "list_areas")
            return [{ id: "area-1", name: "Warehouse area" }];
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
          if (method === "object_targets")
            return {
              runtime: seededRuntime || {
                status: "structurally_available",
                backend: "siglip",
                fingerprint: "test-model",
                structurally_available: true,
                executable_verified: false,
                reason_codes: [],
              },
              targets,
            };
          if (method === "list_zones")
            return [
              {
                name: "Cup shelf polygon",
                polygon: [
                  [0, 0],
                  [1, 0],
                  [1, 1],
                ],
                dwell_alert_seconds: 0,
              },
            ];
          if (method === "create_object_target") {
            const [
              id,
              label,
              category,
              aliases,
              min_similarity,
              allowed_zone_ids,
            ] = args as [string, string, string, string[], number, string[]];
            if (!targets.some((target) => target.id === id)) {
              targets.push({
                id,
                label,
                category,
                aliases,
                min_similarity,
                allowed_zone_ids,
                review_state: "draft",
                examples: [],
                negative_examples: [],
                can_activate: false,
                reasons: ["no_reviewed_positive_examples"],
              });
            }
            return { target: targets.find((target) => target.id === id) };
          }
          if (method === "add_object_example") {
            if (state.failFirstExample) {
              state.failFirstExample = false;
              throw new Error("Example image could not be saved");
            }
            const [objectId, , bbox, source, bbox_format, negative] = args as [
              string,
              string,
              [number, number, number, number],
              string,
              string,
              boolean,
            ];
            const target = targets.find((item) => item.id === objectId);
            const example = {
              id: `ex-${target?.examples.length || 0}`,
              source,
              bbox,
              bbox_format,
              sha256: "test",
              reviewed: false,
            };
            if (negative) target?.negative_examples.push(example);
            else target?.examples.push(example);
            return { target, example };
          }
          if (method === "set_object_watch_rule") return { ok: true };
          if (method === "reembed_object_targets")
            return { job_id: "job-1", status: "queued" };
          if (method === "object_watch_job_status") {
            const next = state.jobStatuses.shift() || {
              job_id: "job-1",
              status: "completed",
              written: 1,
            };
            if (next.status === "completed") {
              for (const target of targets) {
                target.can_activate = true;
                target.reasons = [];
              }
            }
            return next;
          }
          if (method === "object_example_preview")
            return { mime_type: "image/png", image_b64: "AA==" };
          return { ok: true };
        },
        subscribe: () => () => {},
        environment: async () => ({}),
      };
    },
    {
      initialPermissions: permissions,
      failFirstExample: Boolean(options.failFirstExample),
      seededTargets: options.initialTargets || [],
      seededRuntime: options.runtime,
      jobStatuses: options.jobStatuses || [],
    },
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Local engine", exact: true }).click();
  await expect(
    page.getByText("Backend connected", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Object watchlists" }),
  ).toBeVisible();
}

async function expectInsideWatchlist(
  page: import("@playwright/test").Page,
  names: string[],
) {
  const container = await page.locator(".object-watchlist").boundingBox();
  expect(container).toBeTruthy();
  for (const name of names) {
    const item = page.getByRole("button", { name, exact: true }).first();
    await expect(item).toBeVisible();
    const box = await item.boundingBox();
    expect(box, `${name} should have a box`).toBeTruthy();
    expect(box!.x, `${name} left edge`).toBeGreaterThanOrEqual(
      container!.x - 1,
    );
    expect(box!.x + box!.width, `${name} right edge`).toBeLessThanOrEqual(
      container!.x + container!.width + 1,
    );
  }
}

function reviewedDraftTarget(): Target {
  return {
    id: "yellow-cup",
    label: "Yellow cup",
    category: "custom",
    aliases: [],
    review_state: "draft",
    min_similarity: 0.72,
    allowed_zone_ids: [],
    examples: [
      {
        id: "ex-1",
        source: "upload",
        bbox: [0, 0, 10, 10],
        bbox_format: "pixel_xyxy",
        sha256: "test",
        reviewed: true,
      },
    ],
    negative_examples: [],
    can_activate: false,
    reasons: ["no_embeddings"],
  };
}

test("empty library choose photo button opens picker and shows PNG preview", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await mountObjectWatchlist(page, ["view_live", "configure_cameras"]);
  await expectInsideWatchlist(page, ["Choose photo"]);
  await page.screenshot({
    path: "test-results/object-watchlist-empty-1280x800.png",
    fullPage: true,
  });

  const chooserPromise = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Choose photo", exact: true }).click();
  const chooser = await chooserPromise;
  await chooser.setFiles(pngPath);

  await expect(page.getByAltText("Crop argus-mark.png")).toBeVisible();
  await expect(
    page.getByText("argus-mark.png loaded for cropping"),
  ).toBeVisible();
  await expectInsideWatchlist(page, ["Choose photo", "Save photo and example"]);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.screenshot({
    path: "test-results/object-watchlist-preview-1440x900.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 800, height: 800 });
  await expectInsideWatchlist(page, ["Choose photo", "Save photo and example"]);
  await page.screenshot({
    path: "test-results/object-watchlist-preview-800x800.png",
    fullPage: true,
  });
});

test("file picker cancellation, invalid image guidance, and retry", async ({
  page,
}) => {
  await mountObjectWatchlist(page, ["view_live", "configure_cameras"]);

  const cancelled = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Choose photo", exact: true }).click();
  await (await cancelled).setFiles([]);
  await expect(page.getByText(/No image selected/)).toBeVisible();

  const invalid = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Choose photo", exact: true }).click();
  await (
    await invalid
  ).setFiles({
    name: "yellow-cup.heic",
    mimeType: "image/heic",
    buffer: Buffer.from("not a supported browser image"),
  });
  await expect(page.getByText(/HEIC/)).toBeVisible();

  const retry = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Choose photo", exact: true }).click();
  await (await retry).setFiles(pngPath);
  await expect(page.getByAltText("Crop argus-mark.png")).toBeVisible();
});

test("save creates target once, keeps crop after image failure, then reuses target on retry", async ({
  page,
}) => {
  await mountObjectWatchlist(
    page,
    ["view_live", "configure_cameras", "configure_detectors"],
    {
      failFirstExample: true,
    },
  );
  const chooser = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Choose photo", exact: true }).click();
  await (await chooser).setFiles(pngPath);
  await page.getByLabel("Object name", { exact: true }).fill("Yellow cup");
  await page.getByRole("combobox", { name: "Category" }).selectOption("custom");

  await page.getByRole("button", { name: "Save photo and example" }).click();
  await expect(
    page.getByText("Example image could not be saved"),
  ).toBeVisible();
  await expect(page.getByAltText("Crop argus-mark.png")).toBeVisible();

  await page.getByRole("button", { name: "Save photo and example" }).click();
  await expect(page.getByText("Positive example saved")).toBeVisible();
  await expect(
    page
      .locator(".object-target-card")
      .getByText("Yellow cup", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Positive", { exact: true })).toBeVisible();
  await expect(page.getByText("Unreviewed", { exact: true })).toBeVisible();

  const calls = await page.evaluate(
    () => (window as any).__argusObjectTest.invocations,
  );
  const createCalls = calls.filter(
    (call: { method: string }) => call.method === "create_object_target",
  );
  const addCalls = calls.filter(
    (call: { method: string }) => call.method === "add_object_example",
  );
  expect(createCalls).toHaveLength(1);
  expect(addCalls).toHaveLength(2);
  expect(createCalls[0].args.slice(0, 7)).toEqual([
    "yellow-cup",
    "Yellow cup",
    "custom",
    [],
    0.72,
    [],
    "",
  ]);
  expect(addCalls[1].args[0]).toBe("yellow-cup");
  expect(addCalls[1].args[3]).toBe("upload");
  expect(addCalls[1].args[4]).toBe("pixel_xyxy");
  expect(addCalls[1].args[5]).toBe(false);
  expect((addCalls[1].args[2] as number[])[2]).toBeGreaterThan(1);
});

test("camera alert uses polygon zone name, not hierarchy area name", async ({
  page,
}) => {
  await mountObjectWatchlist(
    page,
    ["view_live", "configure_cameras", "configure_detectors"],
    {
      initialTargets: [
        {
          id: "yellow-cup",
          label: "Yellow cup",
          category: "custom",
          aliases: [],
          review_state: "active",
          min_similarity: 0.72,
          allowed_zone_ids: [],
          examples: [],
          negative_examples: [],
          can_activate: true,
          reasons: [],
        },
      ],
    },
  );

  await page
    .getByRole("combobox", { name: "Zone for alert rule" })
    .selectOption("Cup shelf polygon");
  await page.getByRole("button", { name: "Enable object_seen alert" }).click();
  await page.screenshot({
    path: "test-results/object-watchlist-activated-rule-1440x900.png",
    fullPage: true,
  });

  const ruleCall = await page.evaluate(() =>
    (window as any).__argusObjectTest.invocations.find(
      (call: { method: string }) => call.method === "set_object_watch_rule",
    ),
  );
  expect(ruleCall.args).toEqual([
    "camera-001",
    "yellow-cup",
    true,
    "Cup shelf polygon",
  ]);
  expect(ruleCall.args).not.toContain("Warehouse area");
});

test("structurally available runtime can prepare reviewed examples", async ({
  page,
}) => {
  await mountObjectWatchlist(
    page,
    ["view_live", "configure_cameras", "configure_detectors"],
    {
      initialTargets: [reviewedDraftTarget()],
      runtime: {
        status: "structurally_available",
        backend: "siglip",
        fingerprint: "test-model",
        reason_codes: [],
        structurally_available: true,
        executable_verified: false,
      },
      jobStatuses: [
        { job_id: "job-1", status: "running" },
        { job_id: "job-1", status: "completed", written: 1 },
      ],
    },
  );
  await expect(page.getByText("Configured", { exact: true })).toBeVisible();
  const prepare = page.getByRole("button", { name: "Prepare recognition" });
  await expect(prepare).toBeEnabled();
  await expect(
    page.getByRole("button", { name: "Recognize this object" }),
  ).toBeDisabled();

  await page.screenshot({
    path: "test-results/object-watchlist-configured-unprepared.png",
    fullPage: true,
  });
  await prepare.click();
  await expect(page.getByRole("button", { name: /Preparing/ })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Recognize this object" }),
  ).toBeEnabled({ timeout: 5000 });
  await page.screenshot({
    path: "test-results/object-watchlist-prepared-active.png",
    fullPage: true,
  });
  const calls = await page.evaluate(
    () => (window as any).__argusObjectTest.invocations,
  );
  expect(
    calls.some(
      (call: { method: string }) => call.method === "reembed_object_targets",
    ),
  ).toBe(true);
  expect(
    calls.some(
      (call: { method: string }) => call.method === "object_watch_job_status",
    ),
  ).toBe(true);
});

test("unavailable runtime disables prepare and explains why", async ({
  page,
}) => {
  await mountObjectWatchlist(page, ["view_live", "configure_cameras"], {
    initialTargets: [reviewedDraftTarget()],
    runtime: {
      status: "unavailable",
      backend: "siglip",
      fingerprint: null,
      reason_codes: ["missing_local_model"],
      structurally_available: false,
      executable_verified: false,
    },
  });
  await expect(
    page.getByRole("button", { name: "Prepare recognition" }),
  ).toBeDisabled();
  await expect(
    page.getByText("Choose the local semantic model files below."),
  ).toHaveCount(2);
});

test("failed prepare job shows error and keeps activation disabled", async ({
  page,
}) => {
  await mountObjectWatchlist(page, ["view_live", "configure_cameras"], {
    initialTargets: [reviewedDraftTarget()],
    runtime: {
      status: "structurally_available",
      backend: "siglip",
      fingerprint: "test-model",
      reason_codes: [],
      structurally_available: true,
      executable_verified: false,
    },
    jobStatuses: [
      {
        job_id: "job-1",
        status: "failed",
        error: "Could not load local model weights. Check the configured path.",
      },
    ],
  });
  await page.getByRole("button", { name: "Prepare recognition" }).click();
  await expect(
    page.getByText(/Could not load local model weights/),
  ).toBeVisible({
    timeout: 5000,
  });
  await expect(
    page.getByRole("button", { name: "Recognize this object" }),
  ).toBeDisabled();
});

test("operator cannot mutate enrollment or alert rules", async ({ page }) => {
  await mountObjectWatchlist(page, ["view_live"]);
  await expect(page.getByText("Read-only", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Choose photo" })).toHaveCount(
    0,
  );
  await expect(
    page.getByRole("button", { name: "Enable object_seen alert" }),
  ).toHaveCount(0);
  const calls = await page.evaluate(
    () => (window as any).__argusObjectTest.invocations,
  );
  expect(
    calls.some((call: { method: string }) =>
      [
        "create_object_target",
        "add_object_example",
        "set_object_watch_rule",
      ].includes(call.method),
    ),
  ).toBe(false);
});
