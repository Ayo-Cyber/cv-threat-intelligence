import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { describe, expect, it, vi } from "vitest";
import { startOwnedApi } from "../electron/api-supervisor.js";

describe("API readiness request bounds", () => {
  it("uses an abort signal so one readiness request cannot exceed the deadline", async () => {
    const child = new EventEmitter() as any;
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.stdin = new PassThrough();
    child.exitCode = null;
    child.killed = false;
    child.kill = vi.fn(() => true);
    let signal: AbortSignal | undefined;

    await startOwnedApi({
      python: "python",
      root: "/repo",
      site: "site.json",
      db: "events.db",
      port: 8787,
      spawn: (() => child) as any,
      fetch: (async (_url: string, init?: RequestInit) => {
        signal = init?.signal ?? undefined;
        return new Response("ok");
      }) as any,
    });

    expect(signal).toBeInstanceOf(AbortSignal);
  });
});
