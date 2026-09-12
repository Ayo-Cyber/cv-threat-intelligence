import type { StreamDescriptor, Transport } from "./types";

export type ResolvedCameraStream =
  | { kind: "inactive" }
  | { kind: "webrtc"; peer: RTCPeerConnection }
  | { kind: "mjpeg"; url: string; degraded: boolean };

const WHEP_TIMEOUT_MS = 8_000;
const DEFAULT_RETRY_DELAYS_MS = [500, 1_000];

function abortError(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("Aborted", "AbortError");
}

function abortableDelay(delay: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(abortError(signal));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(done, delay);
    function done() {
      signal.removeEventListener("abort", aborted);
      resolve();
    }
    function aborted() {
      clearTimeout(timer);
      signal.removeEventListener("abort", aborted);
      reject(abortError(signal));
    }
    signal.addEventListener("abort", aborted, { once: true });
  });
}

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
  const negotiation = new AbortController();
  const cancelNegotiation = () => negotiation.abort(abortError(signal));
  signal.addEventListener("abort", cancelNegotiation, { once: true });
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    negotiation.abort(new DOMException("WHEP negotiation timed out", "TimeoutError"));
  }, WHEP_TIMEOUT_MS);
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
      signal: negotiation.signal,
    });
    if (!response.ok)
      throw new Error(`WHEP negotiation failed (${response.status})`);
    await peer.setRemoteDescription({
      type: "answer",
      sdp: await response.text(),
    });
    clearTimeout(timeout);
    signal.removeEventListener("abort", cancelNegotiation);
    return peer;
  } catch (error) {
    clearTimeout(timeout);
    signal.removeEventListener("abort", cancelNegotiation);
    signal.removeEventListener("abort", close);
    peer.close();
    if (timedOut) throw new Error("WHEP negotiation timed out");
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
  retryDelaysMs = DEFAULT_RETRY_DELAYS_MS,
  sleep = abortableDelay,
}: {
  cameraId: string;
  active: boolean;
  api: Pick<Transport, "invoke">;
  video: HTMLVideoElement;
  signal: AbortSignal;
  onWebRtcTrack?: () => void;
  connect?: typeof connectWhep;
  retryDelaysMs?: number[];
  sleep?: (milliseconds: number, signal: AbortSignal) => Promise<void>;
}): Promise<ResolvedCameraStream> {
  if (!active) return { kind: "inactive" };
  for (let attempt = 0; ; attempt += 1) {
    try {
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
        if (signal.aborted) throw abortError(signal);
        if (descriptor.mjpeg_fallback)
          return {
            kind: "mjpeg",
            url: descriptor.mjpeg_fallback,
            degraded: true,
          };
        throw error;
      }
    } catch (error) {
      if (signal.aborted) throw abortError(signal);
      const status = (error as { status?: number }).status;
      if (attempt >= retryDelaysMs.length || (status && status !== 503))
        throw error;
      await sleep(retryDelaysMs[attempt], signal);
    }
  }
}
