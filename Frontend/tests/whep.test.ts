import { afterEach, describe, expect, it, vi } from "vitest";
import { connectWhep, resolveCameraStream } from "../src/lib/whep";

class FakePeerConnection {
  static instances: FakePeerConnection[] = [];
  static hangingPhase?:
    "createOffer" | "setLocalDescription" | "setRemoteDescription";
  localDescription: RTCSessionDescriptionInit | null = null;
  remoteDescription: RTCSessionDescriptionInit | null = null;
  closed = false;
  ontrack: RTCPeerConnection["ontrack"] = null;
  addTransceiver = vi.fn();

  constructor() {
    FakePeerConnection.instances.push(this);
  }

  async createOffer() {
    if (FakePeerConnection.hangingPhase === "createOffer")
      return new Promise<RTCSessionDescriptionInit>(() => {});
    return { type: "offer" as const, sdp: "local-offer" };
  }

  async setLocalDescription(value: RTCSessionDescriptionInit) {
    if (FakePeerConnection.hangingPhase === "setLocalDescription")
      return new Promise<void>(() => {});
    this.localDescription = value;
  }

  async setRemoteDescription(value: RTCSessionDescriptionInit) {
    if (FakePeerConnection.hangingPhase === "setRemoteDescription")
      return new Promise<void>(() => {});
    this.remoteDescription = value;
  }

  close() {
    this.closed = true;
  }
}

describe("WHEP streaming", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    FakePeerConnection.instances = [];
    FakePeerConnection.hangingPhase = undefined;
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
        signal: expect.any(AbortSignal),
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

  it("recovers after bounded transient descriptor failures", async () => {
    const unavailable = Object.assign(new Error("engine starting"), {
      status: 503,
    });
    const api = {
      invoke: vi
        .fn()
        .mockRejectedValueOnce(unavailable)
        .mockRejectedValueOnce(unavailable)
        .mockResolvedValueOnce({
          kind: "mjpeg" as const,
          url: "http://127.0.0.1:9000/stream/front?token=x",
        }),
    };
    const sleep = vi.fn(async () => {});

    await expect(
      resolveCameraStream({
        cameraId: "front",
        active: true,
        api,
        video: {} as HTMLVideoElement,
        signal: new AbortController().signal,
        retryDelaysMs: [10, 20],
        sleep,
      }),
    ).resolves.toEqual({
      kind: "mjpeg",
      url: "http://127.0.0.1:9000/stream/front?token=x",
      degraded: false,
    });
    expect(sleep).toHaveBeenCalledTimes(2);
    expect(api.invoke).toHaveBeenCalledTimes(3);
  });

  it("cancels descriptor retry immediately when the tile becomes inactive", async () => {
    const controller = new AbortController();
    const api = {
      invoke: vi.fn(async () => {
        throw Object.assign(new Error("engine starting"), { status: 503 });
      }),
    };
    const sleep = vi.fn(
      (_delay: number, signal: AbortSignal) =>
        new Promise<void>((_resolve, reject) =>
          signal.addEventListener("abort", () => reject(signal.reason), {
            once: true,
          }),
        ),
    );
    const resolving = resolveCameraStream({
      cameraId: "front",
      active: true,
      api,
      video: {} as HTMLVideoElement,
      signal: controller.signal,
      retryDelaysMs: [1000],
      sleep,
    });

    await Promise.resolve();
    controller.abort();

    await expect(resolving).rejects.toMatchObject({ name: "AbortError" });
    expect(api.invoke).toHaveBeenCalledTimes(1);
  });

  it.each([
    "createOffer",
    "setLocalDescription",
    "fetch",
    "response.text",
    "setRemoteDescription",
  ] as const)("times out a hung WHEP %s phase and cleans up", async (phase) => {
    vi.useFakeTimers();
    if (
      phase === "createOffer" ||
      phase === "setLocalDescription" ||
      phase === "setRemoteDescription"
    )
      FakePeerConnection.hangingPhase = phase;
    vi.stubGlobal(
      "fetch",
      phase === "fetch"
        ? vi.fn(() => new Promise<Response>(() => {}))
        : vi.fn(async () => ({
            ok: true,
            status: 201,
            text:
              phase === "response.text"
                ? () => new Promise<string>(() => {})
                : async () => "remote-answer",
          })),
    );
    vi.stubGlobal("RTCPeerConnection", FakePeerConnection);
    const controller = new AbortController();
    const addListener = vi.spyOn(controller.signal, "addEventListener");
    const removeListener = vi.spyOn(controller.signal, "removeEventListener");
    const pending = connectWhep(
      {} as HTMLVideoElement,
      "http://127.0.0.1:1984/api/webrtc?src=front",
      controller.signal,
    );
    const outcomePromise = pending.then(
      () => "resolved",
      (error) => (error as Error).message,
    );

    await vi.advanceTimersByTimeAsync(10_000);
    const outcome = await outcomePromise;

    expect(outcome).toContain("timed out");
    expect(FakePeerConnection.instances[0].closed).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
    expect(removeListener.mock.calls.length).toBe(addListener.mock.calls.length);
  });

  it("parent abort cancels a hung pre-fetch phase and cleans up", async () => {
    vi.useFakeTimers();
    FakePeerConnection.hangingPhase = "createOffer";
    vi.stubGlobal("fetch", vi.fn());
    vi.stubGlobal("RTCPeerConnection", FakePeerConnection);
    const controller = new AbortController();
    const addListener = vi.spyOn(controller.signal, "addEventListener");
    const removeListener = vi.spyOn(controller.signal, "removeEventListener");
    const pending = connectWhep(
      {} as HTMLVideoElement,
      "http://127.0.0.1:1984/api/webrtc?src=front",
      controller.signal,
    );

    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(FakePeerConnection.instances[0].closed).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
    expect(removeListener.mock.calls.length).toBe(
      addListener.mock.calls.length,
    );
  });
});
