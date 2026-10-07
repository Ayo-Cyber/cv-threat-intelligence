import { describe, expect, it } from "vitest";
import { addCameraBlockedReason, missingToAddCamera, cameraTestMessage } from "../src/lib/add-camera";

describe("camera video test result", () => {
  it("rejects legacy RTSP errors even without ok:false", () => {
    expect(() => cameraTestMessage({ error: "Wrong credentials", kind: "auth" })).toThrow("Wrong credentials");
  });
  it("does not mistake a successful port probe for video", () => {
    expect(() => cameraTestMessage({ ok: true })).toThrow("No video frame received");
    expect(() => cameraTestMessage({})).toThrow("Camera connection failed");
  });
  it("reports decoded dimensions on success", () => {
    expect(cameraTestMessage({ ok: true, w: 1280, h: 720 })).toBe("Video received (1280 x 720)");
  });
});

/**
 * 26 Sep: a reachable webcam, a branch and an area selected, and "Add camera"
 * dead — because Camera name was empty and its placeholder ("reception_01")
 * reads as a value. The button refused and said nothing.
 */
describe("why Add camera is disabled", () => {
  const ready = { id: "reception_01", source: "0", placementReady: true };

  it("says nothing when the form is complete", () => {
    expect(addCameraBlockedReason(ready)).toBeNull();
    expect(missingToAddCamera(ready)).toEqual([]);
  });

  it("names the empty camera name — the reported case", () => {
    expect(addCameraBlockedReason({ ...ready, id: "" })).toBe(
      "Still needed before this camera can be added: a camera name.",
    );
  });

  it("treats whitespace as empty, like the button does not", () => {
    expect(missingToAddCamera({ ...ready, id: "   " })).toEqual(["a camera name"]);
  });

  it("names the source, and says what a source looks like", () => {
    const reason = addCameraBlockedReason({ ...ready, source: "" });
    expect(reason).toContain("0 for a webcam");
    expect(reason).toContain("RTSP");
  });

  it("names placement when no area is chosen", () => {
    expect(addCameraBlockedReason({ ...ready, placementReady: false })).toContain(
      "a branch and an area",
    );
  });

  it("lists several missing things readably", () => {
    expect(
      addCameraBlockedReason({ id: "", source: "", placementReady: false }),
    ).toBe(
      "Still needed before this camera can be added: a camera name, " +
        "a camera source (0 for a webcam, or an RTSP address) and a branch and an area.",
    );
  });
});
