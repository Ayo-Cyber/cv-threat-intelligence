import type { StreamDescriptor, Transport } from "./types";

export type ResolvedCameraStream =
  | { kind: "inactive" }
  | { kind: "webrtc"; peer: RTCPeerConnection }
  | { kind: "mjpeg"; url: string; degraded: boolean };

export async function connectWhep(
  video: HTMLVideoElement,
  url: string,
  signal: AbortSignal,
  onTrack?: () => void,
): Promise<RTCPeerConnection> {
  const endpoint = new URL(url);
  if (
    endpoint.protocol !== "http:" ||
    !["127.0.0.1", "localhost", "::1"].includes(endpoint.hostname)
  ) {
    throw new Error(
      "WHEP requires a loopback HTTP endpoint supplied by Argus.",
    );
  }
  if (signal.aborted) throw new DOMException("Aborted", "AbortError");
  const peer = new RTCPeerConnection();
  const close = () => peer.close();
  signal.addEventListener("abort", close, { once: true });
  try {
    peer.addTransceiver("video", { direction: "recvonly" });
    peer.ontrack = (event) => {
      if (signal.aborted) return;
      video.srcObject = event.streams[0] ?? new MediaStream([event.track]);
      void video.play().catch(() => {});
      onTrack?.();
    };
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    const response = await fetch(endpoint.toString(), {
      method: "POST",
      headers: { "content-type": "application/sdp" },
      body: offer.sdp ?? "",
      signal,
    });
    if (!response.ok)
      throw new Error(`WHEP negotiation failed (${response.status})`);
    await peer.setRemoteDescription({
      type: "answer",
      sdp: await response.text(),
    });
    return peer;
  } catch (error) {
    signal.removeEventListener("abort", close);
    peer.close();
    throw error;
  }
}

export async function resolveCameraStream({
  cameraId,
  active,
  api,
  video,
  signal,
  onWebRtcTrack,
  connect = connectWhep,
}: {
  cameraId: string;
  active: boolean;
  api: Pick<Transport, "invoke">;
  video: HTMLVideoElement;
  signal: AbortSignal;
  onWebRtcTrack?: () => void;
  connect?: typeof connectWhep;
}): Promise<ResolvedCameraStream> {
  if (!active) return { kind: "inactive" };
  const descriptor = await api.invoke<StreamDescriptor>("camera_stream", [
    cameraId,
  ]);
  if (descriptor.kind === "mjpeg")
    return { kind: "mjpeg", url: descriptor.url, degraded: false };
  try {
    return {
      kind: "webrtc",
      peer: await connect(video, descriptor.url, signal, onWebRtcTrack),
    };
  } catch (error) {
    if (signal.aborted) throw error;
    if (descriptor.mjpeg_fallback)
      return {
        kind: "mjpeg",
        url: descriptor.mjpeg_fallback,
        degraded: true,
      };
    throw error;
  }
}
