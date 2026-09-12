import { describe, expect, it } from "vitest";
import { cameraStreamPresentation } from "../src/lib/stream-state";

describe("camera stream presentation", () => {
  it("stays inactive when preview is not active", () => {
    expect(
      cameraStreamPresentation({
        active: false,
        cameraState: "connected",
        transport: "none",
        evidence: "none",
        failed: false,
      }),
    ).toEqual({ phase: "inactive", label: "INACTIVE" });
  });

  it("shows reported offline and reconnecting states without a live claim", () => {
    const base = {
      active: true,
      transport: "webrtc" as const,
      evidence: "webrtc-track" as const,
      failed: false,
    };
    expect(
      cameraStreamPresentation({ ...base, cameraState: "offline" }),
    ).toEqual({ phase: "offline", label: "OFFLINE" });
    expect(
      cameraStreamPresentation({ ...base, cameraState: "reconnecting" }),
    ).toEqual({ phase: "degraded", label: "RECONNECTING" });
  });

  it("keeps MJPEG loading until its first image frame", () => {
    const base = {
      active: true,
      cameraState: "connected",
      transport: "mjpeg" as const,
      failed: false,
    };
    expect(cameraStreamPresentation({ ...base, evidence: "none" })).toEqual({
      phase: "loading",
      label: "CONNECTING",
    });
    expect(
      cameraStreamPresentation({ ...base, evidence: "mjpeg-frame" }),
    ).toEqual({ phase: "live", label: "MJPEG FEED" });
  });

  it("keeps WebRTC loading until an ontrack event", () => {
    const base = {
      active: true,
      cameraState: "connected",
      transport: "webrtc" as const,
      failed: false,
    };
    expect(cameraStreamPresentation({ ...base, evidence: "none" })).toEqual({
      phase: "loading",
      label: "CONNECTING",
    });
    expect(
      cameraStreamPresentation({ ...base, evidence: "webrtc-track" }),
    ).toEqual({ phase: "live", label: "WEBRTC FEED" });
  });

  it("labels fallback only after its first MJPEG frame", () => {
    const base = {
      active: true,
      cameraState: "connected",
      transport: "mjpeg-fallback" as const,
      failed: false,
    };
    expect(cameraStreamPresentation({ ...base, evidence: "none" })).toEqual({
      phase: "loading",
      label: "CONNECTING",
    });
    expect(
      cameraStreamPresentation({ ...base, evidence: "mjpeg-frame" }),
    ).toEqual({ phase: "degraded", label: "MJPEG FALLBACK" });
  });
});
