import { afterEach, describe, expect, it, vi } from "vitest";
import { connectWhep, resolveCameraStream } from "../src/lib/whep";

class FakePeerConnection {
  static instances: FakePeerConnection[] = [];
  localDescription: RTCSessionDescriptionInit | null = null;
  remoteDescription: RTCSessionDescriptionInit | null = null;
  closed = false;
  ontrack: RTCPeerConnection["ontrack"] = null;
  addTransceiver = vi.fn();

  constructor() {
    FakePeerConnection.instances.push(this);
  }

  async createOffer() {
    return { type: "offer" as const, sdp: "local-offer" };
  }

  async setLocalDescription(value: RTCSessionDescriptionInit) {
    this.localDescription = value;
  }

  async setRemoteDescription(value: RTCSessionDescriptionInit) {
    this.remoteDescription = value;
  }

  close() {
    this.closed = true;
  }
}

describe("WHEP streaming", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    FakePeerConnection.instances = [];
  });

  it("posts the local offer and applies the WHEP answer", async () => {
    const fetch = vi.fn(
      async () => new Response("remote-answer", { status: 201 }),
    );
    vi.stubGlobal("fetch", fetch);
    vi.stubGlobal("RTCPeerConnection", FakePeerConnection);
    const video = { srcObject: null } as unknown as HTMLVideoElement;
    const controller = new AbortController();

    const peer = await connectWhep(
      video,
      "http://127.0.0.1:1984/api/webrtc?src=front",
      controller.signal,
    );

    expect(FakePeerConnection.instances[0].addTransceiver).toHaveBeenCalledWith(
      "video",
      { direction: "recvonly" },
    );
    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:1984/api/webrtc?src=front",
      expect.objectContaining({
        method: "POST",
        body: "local-offer",
        headers: { "content-type": "application/sdp" },
        signal: controller.signal,
      }),
    );
    expect(peer.remoteDescription).toEqual({
      type: "answer",
      sdp: "remote-answer",
    });
    controller.abort();
    expect(peer.closed).toBe(true);
  });

  it("rejects non-loopback and non-http WHEP endpoints", async () => {
    vi.stubGlobal("RTCPeerConnection", FakePeerConnection);
    const video = { srcObject: null } as unknown as HTMLVideoElement;

    await expect(
      connectWhep(
        video,
        "https://camera.example/api/webrtc",
        new AbortController().signal,
      ),
    ).rejects.toThrow("loopback HTTP");
    expect(FakePeerConnection.instances).toHaveLength(0);
  });

  it("falls back to MJPEG when WHEP negotiation fails", async () => {
    const api = {
      invoke: vi.fn(async () => ({
        kind: "webrtc" as const,
        url: "http://127.0.0.1:1984/api/webrtc?src=front",
        mjpeg_fallback: "http://127.0.0.1:9000/stream/front?token=x",
      })),
    };

    const result = await resolveCameraStream({
      cameraId: "front",
      active: true,
      api,
      video: {} as HTMLVideoElement,
      signal: new AbortController().signal,
      connect: async () => {
        throw new Error("negotiation failed");
      },
    });

    expect(result).toEqual({
      kind: "mjpeg",
      url: "http://127.0.0.1:9000/stream/front?token=x",
      degraded: true,
    });
  });

  it("does not request or negotiate a stream while the tile is inactive", async () => {
    const api = { invoke: vi.fn() };
    const connect = vi.fn();

    const result = await resolveCameraStream({
      cameraId: "front",
      active: false,
      api,
      video: {} as HTMLVideoElement,
      signal: new AbortController().signal,
      connect,
    });

    expect(result).toEqual({ kind: "inactive" });
    expect(api.invoke).not.toHaveBeenCalled();
    expect(connect).not.toHaveBeenCalled();
  });
});
