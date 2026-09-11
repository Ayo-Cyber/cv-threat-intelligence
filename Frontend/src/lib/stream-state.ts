export type StreamTransport = "none" | "webrtc" | "mjpeg" | "mjpeg-fallback";
export type MediaEvidence = "none" | "webrtc-track" | "mjpeg-frame";

export type StreamPresentation = {
  phase: "inactive" | "loading" | "live" | "degraded" | "offline";
  label: string;
};

export function cameraStreamPresentation({
  active,
  cameraState,
  transport,
  evidence,
  failed,
}: {
  active: boolean;
  cameraState?: string;
  transport: StreamTransport;
  evidence: MediaEvidence;
  failed: boolean;
}): StreamPresentation {
  if (!active) return { phase: "inactive", label: "INACTIVE" };

  const health = cameraState?.toLowerCase();
  if (
    failed ||
    health === "offline" ||
    health === "failed" ||
    health === "error"
  )
    return { phase: "offline", label: "OFFLINE" };
  if (health === "reconnecting")
    return { phase: "degraded", label: "RECONNECTING" };
  if (health === "degraded" || health === "stalled")
    return { phase: "degraded", label: "DEGRADED" };

  const hasMedia =
    (transport === "webrtc" && evidence === "webrtc-track") ||
    ((transport === "mjpeg" || transport === "mjpeg-fallback") &&
      evidence === "mjpeg-frame");
  if (!hasMedia) return { phase: "loading", label: "CONNECTING" };
  if (transport === "webrtc") return { phase: "live", label: "WEBRTC FEED" };
  if (transport === "mjpeg") return { phase: "live", label: "MJPEG FEED" };
  return { phase: "degraded", label: "MJPEG FALLBACK" };
}
