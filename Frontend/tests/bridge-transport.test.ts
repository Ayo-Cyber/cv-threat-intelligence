import { describe, expect, it, vi } from "vitest";
import {
  createBridgeTransport,
  registerBridgeStreamProtocol,
} from "../electron/bridge-transport.js";

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

  it("keeps tracking and the publisher token inside the custom protocol", async () => {
    const legacy = vi.fn(async (method: string) =>
      method === "live_start"
        ? { port: 9010, token: "bridge-token" }
        : { ok: true },
    );
    const upstream = vi.fn(async () => new Response("frame"));
    const bridge = createBridgeTransport(legacy, upstream as any);

    const descriptor = await bridge.invoke("camera_stream", ["Front Door", true]);

    expect(descriptor).toEqual({
      kind: "mjpeg",
      url: "argus-stream://camera/Front%20Door?tracking=1",
    });
    expect(JSON.stringify(descriptor)).not.toContain("bridge-token");
    await bridge.load("Front Door", true);
    expect(upstream).toHaveBeenCalledWith(
      "http://127.0.0.1:9010/stream/Front%20Door?tracking=1&token=bridge-token",
    );
  });

  it("parses only the explicit tracking flag at the protocol boundary", async () => {
    let handler!: (request: Request) => Promise<Response> | Response;
    const registrar = {
      handle: vi.fn((_scheme: string, next: typeof handler) => {
        handler = next;
      }),
    };
    const bridge = { load: vi.fn(async () => new Response("frame")) };
    registerBridgeStreamProtocol(registrar, bridge);

    await handler(
      new Request("argus-stream://camera/Front%20Door?tracking=1"),
    );
    await handler(
      new Request("argus-stream://camera/Till?tracking=true"),
    );

    expect(bridge.load).toHaveBeenNthCalledWith(1, "Front Door", true);
    expect(bridge.load).toHaveBeenNthCalledWith(2, "Till", false);
  });
});
