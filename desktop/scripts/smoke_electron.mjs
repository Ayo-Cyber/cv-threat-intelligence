import { _electron as electron } from "@playwright/test";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import assert from "node:assert/strict";
const repo = process.env.ARGUS_REPO;
if (!repo) throw new Error("Set ARGUS_REPO to the existing Python repository.");
const temp = await fs.mkdtemp(path.join(os.tmpdir(), "argus-electron-smoke-"));
const site = path.join(temp, "site.json");
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
  await page.getByRole("button", { name: "Cameras", exact: true }).click();
  await page.getByRole("button", { name: "Add camera", exact: true }).click();
  await page
    .getByLabel("Camera name", { exact: true })
    .fill("native_test_camera");
  await page
    .getByLabel("Camera source", { exact: true })
    .fill(path.join(repo, "data/test_clips/empty_warehouse.mp4"));
  await page
    .getByRole("button", { name: "Add camera", exact: true })
    .last()
    .click();
  await page
    .getByRole("heading", { name: "native_test_camera", exact: true })
    .waitFor();
  const stored = JSON.parse(await fs.readFile(site, "utf8"));
  assert.equal(stored.cameras[0].id, "native_test_camera");
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
    "PASS: production Electron window, isolated owner login, persisted real camera, evidence approval guard and IPC allowlist.",
  );
} finally {
  await app?.close();
  await fs.rm(temp, { recursive: true, force: true });
}
