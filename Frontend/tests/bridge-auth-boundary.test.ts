import { describe, expect, it, vi } from "vitest";
import { createBridgeTransport } from "../electron/bridge-transport.js";

describe("bridge stream auth boundary", () => {
  it("revokes the main-owned stream proxy before signing out", async () => {
    const calls: string[] = [];
    const legacy = vi.fn(async (method: string) => {
      calls.push(method);
      if (method === "live_start") return { port: 9010, token: "secret" };
      return { ok: true };
    });
    const bridge = createBridgeTransport(
      legacy,
      async () => new Response("frame"),
    );
    await bridge.invoke("camera_stream", ["Front"]);

    await bridge.invoke("sign_out", []);

    expect(calls).toEqual(["live_start", "live_stop", "sign_out"]);
    await expect(bridge.load("Front")).rejects.toThrow(
      "Bridge stream is not active",
    );
  });
});
