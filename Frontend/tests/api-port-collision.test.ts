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

const options = {
  python: "python",
  root: "/repo",
  site: "site.json",
  db: "events.db",
  port: 8787,
};

describe("owned API readiness identity", () => {
  it("refuses to spawn when another HTTP service already owns the port", async () => {
    const spawn = vi.fn();

    await expect(
      startOwnedApi({
        ...options,
        spawn: spawn as any,
        fetch: (async () => new Response("other service")) as any,
      }),
    ).rejects.toThrow("port 8787 is already in use");
    expect(spawn).not.toHaveBeenCalled();
  });

  it("rejects a valid probe when the spawned child exits during it", async () => {
    const owned = child();
    const fetch = vi
      .fn()
      .mockRejectedValueOnce(new Error("port free"))
      .mockImplementationOnce(async () => {
        owned.exitCode = 1;
        owned.emit("exit", 1);
        return new Response(
          JSON.stringify({ name: "Argus Engine API", status: "ok" }),
          { status: 200 },
        );
      });

    await expect(
      startOwnedApi({
        ...options,
        spawn: (() => owned) as any,
        fetch: fetch as any,
        sleep: async () => {},
      }),
    ).rejects.toThrow("Argus API exited (code 1)");
  });
});
