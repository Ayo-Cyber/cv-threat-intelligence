import { _electron as electron } from "@playwright/test";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import {
  createSmokeEnvironment,
  fullscreenSkipAllowed,
  waitForSmokeState,
} from "./smoke_api_transport.mjs";
const repo = process.env.ARGUS_REPO;
if (!repo) throw new Error("Set ARGUS_REPO to the existing Python repository.");
const fixture = await createSmokeEnvironment(repo, "electron");
const { env, site, temp } = fixture;
const cameraId = `desktop_zone_smoke_${process.pid}`;

async function nativeFullscreenEscape(page, wall) {
  const capability = await page.evaluate(() => ({
    fullscreenEnabled: document.fullscreenEnabled,
    requestFullscreen:
      typeof Element.prototype.requestFullscreen === "function",
  }));
  if (fullscreenSkipAllowed(capability)) {
    console.warn(
      "SKIP: this environment does not expose the browser Fullscreen API. Manual acceptance: enter Streams wall, enter fullscreen, press Escape, and confirm the wall remains open outside fullscreen.",
    );
    return "skipped";
  }
  assert.equal(
    capability.fullscreenEnabled,
    true,
    "Fullscreen API exists but document.fullscreenEnabled is false",
  );
  const control = wall.getByRole("button", { name: "Enter fullscreen" });
  assert.equal(await control.count(), 1);
  assert.equal(await control.isEnabled(), true);
  await control.click();
  await page.waitForFunction(() => document.fullscreenElement !== null, null, {
    timeout: 5000,
  });
  await page.keyboard.press("Escape");
  await page.waitForFunction(() => document.fullscreenElement === null, null, {
    timeout: 5000,
  });
  assert.equal(await wall.isVisible(), true);
  return "passed";
}

async function rendererContainsToken(page, expectedFingerprint) {
  return page.evaluate(async (expected) => {
    const candidates = new Set();
    const seen = new Set();
    const addString = (value) => {
      candidates.add(value);
      for (const match of value.matchAll(/[A-Za-z0-9_-]{20,}/g))
        candidates.add(match[0]);
      try {
        const parsed = JSON.parse(value);
        if (parsed !== value) addValue(parsed);
      } catch {
        /* Plain storage strings are already included. */
      }
    };
    const addValue = (value) => {
      if (typeof value === "string") return addString(value);
      if (!value || typeof value !== "object" || seen.has(value)) return;
      seen.add(value);
      if (Array.isArray(value)) {
        value.forEach(addValue);
        return;
      }
      for (const [key, item] of Object.entries(value)) {
        addString(key);
        addValue(item);
      }
    };
    for (const storage of [localStorage, sessionStorage]) {
      for (let index = 0; index < storage.length; index += 1) {
        const key = storage.key(index) || "";
        addString(key);
        addString(storage.getItem(key) || "");
      }
    }
    addString(document.cookie);
    if (typeof indexedDB.databases === "function") {
      for (const info of await indexedDB.databases()) {
        if (!info.name) continue;
        addString(info.name);
        const database = await new Promise((resolve, reject) => {
          const request = indexedDB.open(info.name);
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => reject(request.error);
        });
        for (const storeName of Array.from(database.objectStoreNames)) {
          addString(storeName);
          const transaction = database.transaction(storeName, "readonly");
          const store = transaction.objectStore(storeName);
          const read = (request) =>
            new Promise((resolve, reject) => {
              request.onsuccess = () => resolve(request.result);
              request.onerror = () => reject(request.error);
            });
          const [keys, values] = await Promise.all([
            read(store.getAllKeys()),
            read(store.getAll()),
          ]);
          addValue(keys);
          addValue(values);
        }
        database.close();
      }
    }
    const encoder = new TextEncoder();
    for (const candidate of candidates) {
      const digest = await crypto.subtle.digest(
        "SHA-256",
        encoder.encode(candidate),
      );
      const fingerprint = Array.from(new Uint8Array(digest), (byte) =>
        byte.toString(16).padStart(2, "0"),
      ).join("");
      if (fingerprint === expected) return true;
    }
    return false;
  }, expectedFingerprint);
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

  const smokeState = await waitForSmokeState(fixture.smokeState);
  const storageMarkers = [
    "local-storage-smoke-marker-2026",
    "session-storage-smoke-marker-2026",
    "indexed-db-smoke-marker-2026",
  ];
  await page.evaluate(async ([localMarker, sessionMarker, indexedMarker]) => {
    localStorage.setItem("argus.smoke.local", localMarker);
    sessionStorage.setItem("argus.smoke.session", sessionMarker);
    await new Promise((resolve, reject) => {
      const request = indexedDB.open("argus-smoke-token-audit", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("state");
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const database = request.result;
        const transaction = database.transaction("state", "readwrite");
        transaction.objectStore("state").put({ value: indexedMarker }, "token");
        transaction.onerror = () => reject(transaction.error);
        transaction.oncomplete = () => {
          database.close();
          resolve();
        };
      };
    });
  }, storageMarkers);
  for (const marker of storageMarkers) {
    assert.equal(
      await rendererContainsToken(
        page,
        createHash("sha256").update(marker).digest("hex"),
      ),
      true,
      `Renderer storage scanner missed ${marker}`,
    );
  }
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
  assert.deepEqual(exposure.indexedDatabases, ["argus-smoke-token-audit"]);
  assert.equal(
    exposure.storage.flat().some((value) => /bearer|auth.?token/i.test(value)),
    false,
  );
  assert.equal(
    await rendererContainsToken(page, smokeState.token_sha256),
    false,
    "Renderer storage contains the actual Electron-main API token",
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
    `PASS: production Electron, API-default auth, actual-token fingerprint absent from verified local/session/IndexedDB storage, operator permission denial, recovery, real preview, hierarchy wall filters, shell return, IPC allowlist, and native fullscreen Escape ${fullscreenAcceptance}.`,
  );
} finally {
  await app?.close();
  await fixture.cleanup();
}
