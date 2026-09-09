import { _electron as electron } from "@playwright/test";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import assert from "node:assert/strict";
const repo = process.env.ARGUS_REPO;
if (!repo) throw new Error("Set ARGUS_REPO to the existing Python repository.");
const temp = await fs.mkdtemp(path.join(os.tmpdir(), "argus-electron-smoke-"));
const site = path.join(temp, "site.json");
const cameraId = `desktop_zone_smoke_${process.pid}`;
await fs.writeFile(
  site,
  JSON.stringify({
    name: "Electron smoke site",
    notify: "console",
    configured: false,
    cameras: [],
  }),
);
const env = {
  ...process.env,
  ARGUS_REPO: repo,
  ARGUS_SITE_CONFIG: site,
  ARGUS_DB: path.join(temp, "events.db"),
  ARGUS_USER_DATA: path.join(temp, "electron"),
};
delete env.ELECTRON_RUN_AS_NODE;
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
  await page
    .getByRole("button", { name: "Add camera", exact: true })
    .last()
    .click();
  await page.getByRole("heading", { name: cameraId, exact: true }).waitFor();
  const stored = JSON.parse(await fs.readFile(site, "utf8"));
  assert.equal(stored.cameras[0].id, cameraId);
  await page.getByRole("button", { name: "Overview", exact: true }).click();
  await page.getByTitle("Connect camera feeds", { exact: true }).click();
  await page.waitForFunction(() => {
    const image = document.querySelector(".camera-media img");
    return (
      image instanceof HTMLImageElement &&
      image.complete &&
      image.naturalWidth > 0
    );
  });
  await page.getByRole("button", { name: "Cameras", exact: true }).click();
  await page.getByRole("button", { name: "Scene review", exact: true }).click();
  await page
    .getByText("Mapping evidence is missing.", { exact: false })
    .waitFor();
  assert.equal(
    await page
      .getByRole("button", { name: "Approve context", exact: true })
      .isDisabled(),
    true,
  );
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.evaluate(async (id) => {
    await window.argusDesktop.invoke("add_zone", [
      id,
      "Existing backend zone",
      [
        [0, 0],
        [400, 0],
        [400, 300],
        [0, 300],
      ],
      5,
    ]);
  }, cameraId);
  await page.getByRole("tab", { name: "Zones", exact: true }).click();
  await page
    .getByRole("button", { name: "Remove Existing backend zone" })
    .waitFor({ timeout: 30000 });
  assert.equal(
    await page.locator(".saved-zone").getAttribute("points"),
    "0,0 400,0 400,300 0,300",
  );
  await page.screenshot({ path: "test-results/electron-native-zones.png" });
  await page
    .getByRole("button", { name: "Remove Existing backend zone" })
    .click();
  await page.waitForFunction(
    async (id) =>
      (await window.argusDesktop.invoke("list_zones", [id])).length === 0,
    cameraId,
  );
  assert.deepEqual(pageErrors, []);
  const blocked = await page.evaluate(async () => {
    try {
      await window.argusDesktop.invoke("__getattribute__", ["accounts"]);
      return false;
    } catch {
      return true;
    }
  });
  assert.equal(blocked, true);
  await page.screenshot({ path: "test-results/electron-native.png" });
  console.log(
    "PASS: production Electron, owner creates operator, operator login and permission denial, recovery command, real preview, full-screen backend zones, evidence guard and IPC allowlist.",
  );
} finally {
  await app?.close();
  for (const directory of ["zones", "rules"]) {
    await fs.rm(path.join(repo, "configs", directory, `${cameraId}.json`), {
      force: true,
    });
  }
  await fs.rm(temp, { recursive: true, force: true });
}
