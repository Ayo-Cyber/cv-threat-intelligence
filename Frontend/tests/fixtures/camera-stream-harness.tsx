import { createRoot } from "react-dom/client";
import CameraStream from "../../src/components/CameraStream";
import type { StreamDescriptor, Transport } from "../../src/lib/types";

let descriptor: StreamDescriptor;
let failOffer = false;
let latestPeer: FakePeer | undefined;

class FakePeer {
  ontrack: RTCPeerConnection["ontrack"] = null;
  localDescription: RTCSessionDescriptionInit | null = null;
  constructor() {
    latestPeer = this;
  }
  addTransceiver() {}
  async createOffer() {
    if (failOffer) throw new Error("WHEP unavailable");
    return { type: "offer" as const, sdp: "offer" };
  }
  async setLocalDescription(value: RTCSessionDescriptionInit) {
    this.localDescription = value;
  }
  async setRemoteDescription() {}
  close() {}
}

window.RTCPeerConnection = FakePeer as any;
const transport: Transport = {
  async invoke<T>() {
    return descriptor as T;
  },
};
const root = createRoot(document.getElementById("root")!);

(window as any).cameraHarness = {
  render(options: {
    state: string;
    descriptor: StreamDescriptor;
    active?: boolean;
    failWhep?: boolean;
  }) {
    descriptor = options.descriptor;
    failOffer = options.failWhep ?? false;
    latestPeer = undefined;
    root.render(
      <CameraStream
        camera={{
          id: "Front Door",
          source: "rtsp://front",
          state: options.state,
        }}
        api={transport}
        active={options.active ?? true}
      />,
    );
  },
  emitTrack() {
    if (!latestPeer?.ontrack) return false;
    const stream = new MediaStream();
    latestPeer.ontrack({
      streams: [stream],
      track: {} as MediaStreamTrack,
    } as RTCTrackEvent);
    return true;
  },
};
