import { describe, expect, it, vi } from "vitest";
import { createBridgeTransport } from "../electron/bridge-transport.js";

describe("explicit bridge rollback transport", () => {
  it("adapts one legacy live publisher into per-camera MJPEG descriptors", async () => {
    const legacy = vi.fn(async (method: string) => {
      if (method === "live_start") return { port: 9010, token: "bridge-token" };
      return { ok: true };
    });
    const upstream = vi.fn(async () => new Response("frame"));
    const bridge = createBridgeTransport(legacy, upstream as any);

    const first = await bridge.invoke("camera_stream", ["Front Door"]);
    const second = await bridge.invoke("camera_stream", ["Till"]);

    expect(first).toEqual({
      kind: "mjpeg",
      url: "argus-stream://camera/Front%20Door",
    });
    expect(second).toEqual({
      kind: "mjpeg",
      url: "argus-stream://camera/Till",
    });
    expect(JSON.stringify(first)).not.toContain("bridge-token");
    await bridge.load("Front Door");
    expect(upstream).toHaveBeenCalledWith(
      "http://127.0.0.1:9010/stream/Front%20Door?token=bridge-token",
    );
    expect(legacy).toHaveBeenCalledTimes(1);
  });
});
