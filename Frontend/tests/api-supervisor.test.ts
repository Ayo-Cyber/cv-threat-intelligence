import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { describe, expect, it, vi } from "vitest";
import { startOwnedApi } from "../electron/api-supervisor.js";

function child() {
  const process = new EventEmitter() as any;
  process.stdout = new PassThrough();
  process.stderr = new PassThrough();
  process.stdin = new PassThrough();
  process.exitCode = null;
  process.killed = false;
  process.kill = vi.fn(() => {
    process.killed = true;
    return true;
  });
  return process;
}

describe("owned API supervision", () => {
  it("spawns the requested loopback API and waits until its root is ready", async () => {
    const owned = child();
    const spawn = vi.fn(() => owned);
    const fetch = vi
      .fn()
      .mockRejectedValueOnce(new Error("port free"))
      .mockRejectedValueOnce(new Error("starting"))
      .mockResolvedValueOnce(
        new Response('{"name":"Argus Engine API","status":"ok"}', {
          status: 200,
        }),
      );
    const sleep = vi.fn(async () => {});

    const api = await startOwnedApi({
      python: "/repo/.venv/bin/python",
      root: "/repo",
      site: "configs/site_live.json",
      db: "runs/desktop/events.db",
      port: 8787,
      spawn: spawn as any,
      fetch: fetch as any,
      sleep,
      timeoutMs: 30_000,
    });

    expect(spawn).toHaveBeenCalledWith(
      "/repo/.venv/bin/python",
      [
        "-u",
        "-m",
        "cvti.api",
        "--host",
        "127.0.0.1",
        "--port",
        "8787",
        "--site",
        "configs/site_live.json",
        "--db",
        "runs/desktop/events.db",
      ],
      expect.objectContaining({ cwd: "/repo", stdio: "pipe" }),
    );
    expect(fetch).toHaveBeenCalledTimes(3);
    expect(sleep).toHaveBeenCalledWith(250);
    expect(api.baseUrl).toBe("http://127.0.0.1:8787/api/v1");

    await api.stop();
    expect(owned.kill).toHaveBeenCalledWith("SIGTERM");
  });

  it("fails after the readiness deadline and terminates only its child", async () => {
    const owned = child();
    const unrelated = child();
    let now = 0;

    await expect(
      startOwnedApi({
        python: "python",
        root: "/repo",
        site: "site.json",
        db: "events.db",
        port: 9000,
        spawn: (() => owned) as any,
        fetch: (async () => {
          throw new Error("not ready");
        }) as any,
        sleep: async (delay) => {
          now += delay;
        },
        now: () => now,
        timeoutMs: 500,
      }),
    ).rejects.toThrow("did not become ready within 0.5 seconds");
    expect(owned.kill).toHaveBeenCalledWith("SIGTERM");
    expect(unrelated.kill).not.toHaveBeenCalled();
  });
});
