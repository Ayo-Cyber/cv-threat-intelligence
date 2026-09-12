import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const frontend = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(frontend, "../..");
let api: ChildProcess | undefined;
let apiBase = "";
let tempDirectory = "";
let installerToken = "";
let apiOutput = "";

async function reservePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close();
        reject(new Error("could not reserve an API port"));
        return;
      }
      server.close((error) => (error ? reject(error) : resolve(address.port)));
    });
  });
}

async function request(
  pathName: string,
  body: object,
  token?: string,
): Promise<any> {
  const response = await fetch(`${apiBase}${pathName}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
  });
  if (!response.ok)
    throw new Error(
      `${pathName} failed (${response.status}): ${await response.text()}`,
    );
  return response.json();
}

test.beforeAll(async () => {
  tempDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "argus-ws-auth-"));
  const site = path.join(tempDirectory, "site.json");
  fs.writeFileSync(site, '{"name":"ws-auth-test","cameras":[]}');
  const port = await reservePort();
  apiBase = `http://127.0.0.1:${port}/api/v1`;
  api = spawn(
    path.join(root, ".venv/bin/python"),
    [
      "-m",
      "cvti.api",
      "--host",
      "127.0.0.1",
      "--port",
      String(port),
      "--db",
      path.join(tempDirectory, "events.db"),
      "--site",
      site,
    ],
    {
      cwd: root,
      env: { ...process.env, PYTHONPYCACHEPREFIX: tempDirectory },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  api.stdout?.on("data", (chunk) => (apiOutput += String(chunk)));
  api.stderr?.on("data", (chunk) => (apiOutput += String(chunk)));

  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${apiBase}/auth/state`);
      if (response.ok) break;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  const owner = await request("/auth/first-owner", {
    username: "owner",
    password: "owner-password-123",
  });
  expect(owner.ok).toBe(true);
  const ownerSession = await request("/auth/session", {
    username: "owner",
    password: "owner-password-123",
  });
  await request(
    "/users",
    {
      username: "installer",
      password: "installer-password-123",
      role: "installer",
    },
    ownerSession.token,
  );
  const installerSession = await request("/auth/session", {
    username: "installer",
    password: "installer-password-123",
  });
  installerToken = installerSession.token;
});

test.afterAll(async () => {
  if (api && api.exitCode === null) {
    api.kill("SIGTERM");
    await Promise.race([
      new Promise((resolve) => api?.once("exit", resolve)),
      new Promise((resolve) => setTimeout(resolve, 5_000)),
    ]);
    if (api.exitCode === null) api.kill("SIGKILL");
  }
  if (tempDirectory) fs.rmSync(tempDirectory, { recursive: true, force: true });
});

test("installer denial is terminal and emits no alert data", async ({
  page,
}) => {
  await page.goto("/");
  const result = await page.evaluate(
    async ({ baseUrl, token }) => {
      const { ArgusApiClient } = await import("/electron/api-client.ts");
      let connections = 0;
      const messages: string[] = [];
      let resolveClose!: (value: { code: number; protocol: string }) => void;
      const firstClose = new Promise<{ code: number; protocol: string }>(
        (resolve) => (resolveClose = resolve),
      );
      class CountingWebSocket extends WebSocket {
        constructor(url: string | URL, protocols?: string | string[]) {
          super(url, protocols);
          connections += 1;
          this.addEventListener("message", (event) =>
            messages.push(String(event.data)),
          );
          this.addEventListener(
            "close",
            (event) =>
              resolveClose({ code: event.code, protocol: this.protocol }),
            { once: true },
          );
        }
      }
      const client = new ArgusApiClient(baseUrl, {
        WebSocket: CountingWebSocket as typeof WebSocket,
      });
      Object.defineProperty(client, "token", { value: token, writable: true });
      client.subscribe(() => {});
      const closed = await Promise.race([
        firstClose,
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("websocket did not close")), 5_000),
        ),
      ]);
      await new Promise((resolve) => setTimeout(resolve, 1_500));
      await client.close();
      return { ...closed, connections, messages };
    },
    { baseUrl: apiBase, token: installerToken },
  );

  expect(result, apiOutput).toEqual({
    code: 4403,
    protocol: "argus.v1",
    connections: 1,
    messages: [],
  });
});
