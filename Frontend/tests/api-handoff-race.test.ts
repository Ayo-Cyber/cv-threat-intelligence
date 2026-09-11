import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { describe, expect, it, vi } from "vitest";
import { createOwnedApiClient } from "../electron/api-runtime.js";
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

describe("owned API readiness handoff", () => {
  it("blocks authentication when the owned child exits after readiness", async () => {
    const owned = child();
    const fetch = vi
      .fn()
      .mockRejectedValueOnce(new Error("port free"))
      .mockResolvedValueOnce(
        new Response('{"name":"Argus Engine API","status":"ok"}'),
      )
      .mockResolvedValueOnce(
        new Response(
          '{"token":"wrong-process-token","user":{"username":"ayo","role":"owner","permissions":[]}}',
        ),
      );
    const api = await startOwnedApi({
      python: "python",
      root: "/repo",
      site: "site.json",
      db: "events.db",
      port: 8787,
      spawn: (() => owned) as any,
      fetch: fetch as any,
      sleep: async () => {},
    });
    const exited = vi.fn();
    api.onExit(exited);
    const client = createOwnedApiClient(api);

    owned.exitCode = 1;
    owned.emit("exit", 1);

    await expect(client.invoke("sign_in", ["ayo", "secret"])).rejects.toThrow(
      "Argus API exited (code 1)",
    );
    expect(exited).toHaveBeenCalledOnce();
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});
