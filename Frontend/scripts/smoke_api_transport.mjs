import { _electron as electron } from "@playwright/test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import http from "node:http";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

export async function reserveLoopbackPort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const port = address.port;
  await new Promise((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
  return port;
}

export async function createSmokeEnvironment(repo, label = "api") {
  const temp = await fs.mkdtemp(
    path.join(os.tmpdir(), `argus-${label}-smoke-`),
  );
  const site = path.join(temp, "site.json");
  const db = path.join(temp, "events.db");
  const userData = path.join(temp, "electron");
  const source = path.join(repo, "data/test_clips/empty_warehouse.mp4");
  const apiPort = await reserveLoopbackPort();
  const pixel = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
    "base64",
  );
  const streamServer = http.createServer((_request, response) => {
    response.writeHead(200, {
      "cache-control": "no-store",
      "content-length": pixel.length,
      "content-type": "image/png",
    });
    response.end(pixel);
  });
  await new Promise((resolve, reject) => {
    streamServer.once("error", reject);
    streamServer.listen(0, "127.0.0.1", resolve);
  });
  const streamAddress = streamServer.address();
  assert.ok(streamAddress && typeof streamAddress === "object");
  const streamPort = streamAddress.port;
  const fixture = {
    name: "Argus Smoke Site",
    notify: "console",
    configured: false,
    organization: { id: "argus-smoke", name: "Argus Smoke Organization" },
    branches: [
      { id: "west", name: "West Branch" },
      { id: "east", name: "East Branch" },
    ],
    areas: [
      { id: "lobby", name: "Lobby", branch_id: "west" },
      { id: "loading", name: "Loading", branch_id: "west" },
      { id: "warehouse", name: "Warehouse", branch_id: "east" },
    ],
    cameras: [
      { id: "Lobby North", source, area_id: "lobby" },
      { id: "Lobby South", source, area_id: "lobby" },
      { id: "Loading Bay", source, area_id: "loading" },
      { id: "Warehouse A", source, area_id: "warehouse" },
    ],
  };
  await fs.writeFile(site, JSON.stringify(fixture, null, 2));
  await fs.writeFile(
    path.join(temp, "frames.json"),
    JSON.stringify({ port: streamPort, token: "temporary-smoke-stream-token" }),
  );
  const env = {
    ...process.env,
    ARGUS_REPO: repo,
    ARGUS_API_PORT: String(apiPort),
    ARGUS_SITE_CONFIG: site,
    ARGUS_DB: db,
    ARGUS_USER_DATA: userData,
    PYTHONDONTWRITEBYTECODE: "1",
    PYTHONPYCACHEPREFIX: path.join(temp, "pycache"),
  };
  delete env.ELECTRON_RUN_AS_NODE;
  return {
    apiPort,
    db,
    env,
    site,
    streamUrl: `http://127.0.0.1:${streamPort}/stream/Lobby%20North`,
    temp,
    userData,
    async cleanup() {
      await new Promise((resolve, reject) =>
        streamServer.close((error) => (error ? reject(error) : resolve())),
      );
      await fs.rm(temp, { recursive: true, force: true });
    },
  };
}

async function waitForApiShutdown(baseUrl, timeoutMs = 15_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      await fetch(baseUrl, { signal: AbortSignal.timeout(500) });
    } catch {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Electron-owned API remained reachable at ${baseUrl}`);
}

async function run() {
  const repo = process.env.ARGUS_REPO;
  if (!repo)
    throw new Error("Set ARGUS_REPO to the existing Python repository.");
  const fixture = await createSmokeEnvironment(repo, "api-transport");
  const baseUrl = `http://127.0.0.1:${fixture.apiPort}/api/v1`;
  let app;
  let monitoringStarted = false;
  try {
    app = await electron.launch({
      args: [".", "--production"],
      env: fixture.env,
    });
    const page = await app.firstWindow();
    await page
      .getByRole("heading", { name: "Every camera. One clear picture." })
      .waitFor();
    await page
      .getByRole("button", { name: "Local engine", exact: true })
      .click();
    await page
      .getByRole("heading", { name: "Make this workspace yours." })
      .waitFor({ timeout: 45_000 });

    const discovery = await fetch(baseUrl).then((response) => response.json());
    assert.equal(discovery.name, "Argus Engine API");
    assert.equal(discovery.status, "ok");
    assert.ok(discovery.endpoints.includes("/api/v1/cameras/discovery"));

    const invoke = (method, args = []) =>
      page.evaluate(
        ({ method: operation, args: operationArgs }) =>
          window.argusDesktop.invoke(operation, operationArgs),
        { method, args },
      );
    const initialAuth = await invoke("auth_state");
    assert.equal(initialAuth.signed_in, false);
    await invoke("create_first_owner", [
      "desktop_smoke",
      "Desktop-Smoke-Only-2026!",
    ]);
    const auth = await invoke("auth_state");
    assert.equal(auth.signed_in, true);
    assert.equal(auth.username, "desktop_smoke");
    assert.equal(auth.role, "owner");

    const subnet = await invoke("detect_subnet");
    assert.ok(Object.hasOwn(subnet, "cidr"));
    const cameras = await invoke("list_cameras");
    assert.deepEqual(
      cameras.map((camera) => camera.id),
      ["Lobby North", "Lobby South", "Loading Bay", "Warehouse A"],
    );
    const hierarchy = await invoke("hierarchy");
    assert.equal(hierarchy.organization.id, "argus-smoke");
    assert.equal(hierarchy.branches.length, 2);
    assert.equal(
      hierarchy.branches.flatMap((branch) => branch.areas).length,
      3,
    );
    const scene = await invoke("scene_context", ["Lobby North"]);
    assert.ok(scene === null || typeof scene === "object");
    const branches = await invoke("create_branch", [
      { id: "north", name: "North Branch" },
    ]);
    assert.ok(branches.some((branch) => branch.id === "north"));
    const hierarchyAfterWrite = await invoke("hierarchy");
    assert.ok(
      hierarchyAfterWrite.branches.some((branch) => branch.id === "north"),
    );

    const descriptor = await invoke("camera_stream", ["Lobby North"]);
    assert.equal(descriptor.kind, "mjpeg");
    assert.match(
      descriptor.url,
      /^http:\/\/127\.0\.0\.1:\d+\/stream\/Lobby North\?token=/,
    );

    const started = await invoke("start_monitoring");
    assert.equal(started.running, true);
    monitoringStarted = true;
    const stopped = await invoke("stop_monitoring");
    assert.equal(stopped.running, false);
    monitoringStarted = false;
  } finally {
    if (app && monitoringStarted) {
      const page = await app.firstWindow().catch(() => undefined);
      await page
        ?.evaluate(() => window.argusDesktop.invoke("stop_monitoring", []))
        .catch(() => undefined);
    }
    await app?.close();
    if (app) await waitForApiShutdown(baseUrl);
    await fixture.cleanup();
  }
  console.log(
    "PASS: Electron-owned API discovery, auth, hierarchy read/write, camera and scene reads, stream descriptor, monitoring commands, and child shutdown.",
  );
}

if (
  process.argv[1] &&
  pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url
)
  await run();
