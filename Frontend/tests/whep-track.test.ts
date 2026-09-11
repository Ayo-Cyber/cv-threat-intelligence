import { afterEach, describe, expect, it, vi } from "vitest";
import { connectWhep } from "../src/lib/whep";

class Peer {
  static instance: Peer;
  localDescription: RTCSessionDescriptionInit | null = null;
  ontrack: RTCPeerConnection["ontrack"] = null;
  constructor() {
    Peer.instance = this;
  }
  addTransceiver() {}
  async createOffer() {
    return { type: "offer" as const, sdp: "offer" };
  }
  async setLocalDescription(value: RTCSessionDescriptionInit) {
    this.localDescription = value;
  }
  async setRemoteDescription() {}
  close() {}
}

describe("WHEP media evidence", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("reports readiness only when the peer emits an ontrack event", async () => {
    vi.stubGlobal("RTCPeerConnection", Peer);
    vi.stubGlobal("fetch", async () => new Response("answer"));
    const play = vi.fn(async () => {});
    const video = { srcObject: null, play } as unknown as HTMLVideoElement;
    const ready = vi.fn();
    await connectWhep(
      video,
      "http://127.0.0.1:1984/api/webrtc?src=front",
      new AbortController().signal,
      ready,
    );

    expect(ready).not.toHaveBeenCalled();
    const stream = {} as MediaStream;
    Peer.instance.ontrack?.({
      streams: [stream],
      track: {} as MediaStreamTrack,
    } as RTCTrackEvent);
    expect(video.srcObject).toBe(stream);
    expect(ready).toHaveBeenCalledOnce();
  });
});
