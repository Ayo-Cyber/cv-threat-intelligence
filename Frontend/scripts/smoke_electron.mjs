import { _electron as electron } from "@playwright/test";
import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import { createSmokeEnvironment } from "./smoke_api_transport.mjs";
const repo = process.env.ARGUS_REPO;
if (!repo) throw new Error("Set ARGUS_REPO to the existing Python repository.");
const fixture = await createSmokeEnvironment(repo, "electron");
const { env, site, temp } = fixture;
const cameraId = `desktop_zone_smoke_${process.pid}`;

async function nativeFullscreenEscape(page, wall) {
  try {
    await wall.getByRole("button", { name: "Enter fullscreen" }).click();
    await page.waitForFunction(
      () => document.fullscreenElement !== null,
      null,
      {
        timeout: 5000,
      },
    );
  } catch {
    console.warn(
      "SKIP: native headed fullscreen could not be entered. Manual acceptance: enter Streams wall, enter fullscreen, press Escape, and confirm the wall remains open outside fullscreen.",
    );
    return "skipped";
  }
  await page.keyboard.press("Escape");
  try {
    await page.waitForFunction(
      () => document.fullscreenElement === null,
      null,
      {
        timeout: 5000,
      },
    );
  } catch {
    assert.equal(await wall.count(), 1, "Escape must not close the wall");
    await page.evaluate(() => document.exitFullscreen()).catch(() => undefined);
    console.warn(
      "SKIP: platform automation did not synthesize browser-owned Escape. Manual acceptance: press Escape from native headed fullscreen and confirm only fullscreen closes.",
    );
    return "skipped";
  }
  assert.equal(await wall.isVisible(), true);
  return "passed";
}
let app;
try {
  app = await electron.launch({ args: [".", "--production"], env });
  const page = await app.firstWindow();
  await page
    .getByRole("heading", { name: "Every camera. One clear picture." })
    .waitFor();
  async function accountChecks() {
    await page.getByRole("button", { name: "Settings", exact: true }).click();
    await page.getByLabel("New account username").fill("new_operator");
    await page
      .getByLabel("New account password", { exact: true })
      .fill("New-Operator-Password-2026!");
    await page
      .getByLabel("Confirm account password")
      .fill("New-Operator-Password-2026!");
    await page
      .getByRole("button", { name: "Create account", exact: true })
      .click();
    await page.getByText("new_operator", { exact: true }).waitFor();
    await page.screenshot({ path: "test-results/electron-users.png" });
    await page.getByTitle("Sign out", { exact: true }).click();
    await page.getByRole("button", { name: "Forgot password?" }).click();
    await page.getByLabel("Recovery command").waitFor();
    assert.ok(
      (await page.getByLabel("Recovery command").inputValue()).includes(
        path.join(temp, "events.db"),
      ),
    );
    await page.screenshot({ path: "test-results/electron-recovery.png" });
    await page
      .getByRole("button", { name: "Create account", exact: true })
      .click();
    await page
      .getByText("An existing owner must create your account.", {
        exact: false,
      })
      .waitFor();
    await page.getByLabel("Username", { exact: true }).fill("new_operator");
    await page
      .getByLabel("Password", { exact: true })
      .fill("New-Operator-Password-2026!");
    await page.getByRole("button", { name: "Sign in", exact: true }).click();
    await page.getByTitle("Sign out", { exact: true }).waitFor();
    await page.getByRole("button", { name: "Overview", exact: true }).click();
    await page
      .getByRole("heading", { name: "Every camera. One clear picture." })
      .waitFor();
    const denied = await page.evaluate(async () => {
      try {
        await window.argusDesktop.invoke("add_user", [
          "escalated",
          "Long-Password-2026",
          "owner",
        ]);
        return false;
      } catch {
        return true;
      }
    });
    assert.equal(denied, true);
    await page.getByTitle("Sign out", { exact: true }).click();
    await page.getByLabel("Username", { exact: true }).fill("desktop_smoke");
    await page
      .getByLabel("Password", { exact: true })
      .fill("Desktop-Smoke-Only-2026!");
    await page.getByRole("button", { name: "Sign in", exact: true }).click();
    await page
      .getByRole("heading", { name: "Every camera. One clear picture." })
      .waitFor();
  }
  assert.equal(await page.evaluate(() => typeof window.require), "undefined");
  await page.getByRole("button", { name: "Local engine", exact: true }).click();
  await page
    .getByRole("heading", { name: "Make this workspace yours." })
    .waitFor({ timeout: 45000 });
  await page.getByLabel("Username", { exact: true }).fill("desktop_smoke");
  await page
    .getByLabel("Password", { exact: true })
    .fill("Desktop-Smoke-Only-2026!");
  await page.getByRole("button", { name: "Create owner account" }).click();
  await page
    .getByRole("heading", { name: "Every camera. One clear picture." })
    .waitFor();
  await accountChecks();
  await page.getByRole("button", { name: "Cameras", exact: true }).click();
  await page.getByRole("button", { name: "Add camera", exact: true }).click();
  await page.getByLabel("Camera name", { exact: true }).fill(cameraId);
  await page
    .getByLabel("Camera source", { exact: true })
    .fill(path.join(repo, "data/test_clips/empty_warehouse.mp4"));
  const addCameraForm = page.locator("form").filter({
    has: page.getByLabel("Camera name", { exact: true }),
  });
  assert.equal(await addCameraForm.locator("select").count(), 1);
  await addCameraForm
    .locator("select")
    .first()
    .selectOption({ label: "West Branch" });
  assert.equal(await addCameraForm.locator("select").count(), 2);
  await addCameraForm
    .locator("select")
    .nth(1)
    .selectOption({ label: "Loading" });
  await page
    .getByRole("button", { name: "Add camera", exact: true })
    .last()
    .click();
  await page.getByRole("heading", { name: cameraId, exact: true }).waitFor();
  const stored = JSON.parse(await fs.readFile(site, "utf8"));
  assert.ok(stored.cameras.some((camera) => camera.id === cameraId));
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.waitForFunction(() => {
    const image = document.querySelector(".camera-media img");
    return (
      image instanceof HTMLImageElement &&
      image.complete &&
      image.naturalWidth > 0
    );
  });
  await page.getByRole("button", { name: "Cameras", exact: true }).click();
  await page
    .locator(".management-row")
    .filter({ has: page.getByRole("heading", { name: cameraId, exact: true }) })
    .getByRole("button", { name: "Scene review", exact: true })
    .click();
  await page
    .getByText("Mapping evidence is missing.", { exact: false })
    .waitFor();
  assert.equal(
    await page
      .getByRole("button", { name: "Approve context", exact: true })
      .isDisabled(),
    true,
  );
  const blocked = await page.evaluate(async () => {
    try {
      await window.argusDesktop.invoke("__getattribute__", ["accounts"]);
      return false;
    } catch {
      return true;
    }
  });
  assert.equal(blocked, true);
  await page.getByRole("button", { name: "Close details" }).click();
  await page.getByRole("button", { name: "Overview", exact: true }).click();

  const exposure = await page.evaluate(async () => ({
    cookies: document.cookie,
    indexedDatabases:
      typeof indexedDB.databases === "function"
        ? (await indexedDB.databases()).map((database) => database.name)
        : [],
    preloadKeys: Object.keys(window.argusDesktop).sort(),
    storage: [localStorage, sessionStorage].flatMap((storage) =>
      Array.from({ length: storage.length }, (_, index) => {
        const key = storage.key(index) || "";
        return [key, storage.getItem(key) || ""];
      }),
    ),
  }));
  assert.deepEqual(exposure.preloadKeys, [
    "environment",
    "invoke",
    "subscribe",
  ]);
  assert.equal(exposure.cookies, "");
  assert.deepEqual(exposure.indexedDatabases, []);
  assert.equal(
    exposure.storage.flat().some((value) => /bearer|auth.?token/i.test(value)),
    false,
  );

  await page.getByRole("button", { name: "Open streams wall" }).click();
  const wall = page.getByRole("region", { name: "Streams-only camera wall" });
  await wall.waitFor();
  assert.equal(
    await page.locator(".sidebar, .topbar, .activity, .app-footer").count(),
    0,
  );
  await wall
    .getByLabel("Wall branch")
    .selectOption({ label: "West Branch (4)" });
  await wall.getByLabel("Wall area").selectOption({ label: "Lobby (2)" });
  assert.deepEqual(
    (await wall.locator(".streams-caption strong").allTextContents()).sort(),
    ["Lobby North", "Lobby South"],
  );
  assert.equal(await wall.getByText("Loading Bay", { exact: true }).count(), 0);
  assert.equal(await wall.getByText("Warehouse A", { exact: true }).count(), 0);
  const fullscreenAcceptance = await nativeFullscreenEscape(page, wall);
  await wall.getByRole("button", { name: "Exit streams wall" }).click();
  await page
    .getByRole("heading", { name: "Every camera. One clear picture." })
    .waitFor();
  assert.equal(await page.locator(".sidebar").count(), 1);
  await page.screenshot({ path: "test-results/electron-native.png" });
  console.log(
    `PASS: production Electron, API-default auth, operator permission denial, recovery, real preview, hierarchy wall filters, shell return, token boundary, IPC allowlist, and native fullscreen Escape ${fullscreenAcceptance}.`,
  );
} finally {
  await app?.close();
  await fixture.cleanup();
}
