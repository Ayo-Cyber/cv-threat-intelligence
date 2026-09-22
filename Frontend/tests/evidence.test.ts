import { describe, expect, it } from "vitest";
import {
  clipRequestKey,
  evidenceView,
  playbackProblem,
  PENDING_WINDOW_S,
} from "../src/lib/evidence";

const now = 1_700_000_000_000;
const fresh = { ts: now / 1000 - 5 };
const old = { ts: now / 1000 - PENDING_WINDOW_S - 5 };

describe("clipRequestKey", () => {
  it("asks by event id, never by evidence path", () => {
    expect(clipRequestKey({ id: "evt_7" })).toBe("evt_7");
    expect(clipRequestKey({ id: 7 })).toBe("7");
  });
});

describe("evidenceView", () => {
  it("plays the video when there is one and it decodes", () => {
    expect(evidenceView({ uri: "data:video/mp4;base64,AA==" }, fresh, { now })).toBe(
      "video",
    );
  });

  it("falls back to the frames when the video cannot be decoded", () => {
    const clip = { uri: "data:video/mp4;base64,AA==", frames: ["data:image/jpeg;base64,/9k="] };
    expect(evidenceView(clip, fresh, { videoFailed: true, now })).toBe("frames");
  });

  it("is honest when the video fails and no frames exist", () => {
    const clip = { uri: "data:video/mp4;base64,AA==", frames: [] };
    expect(evidenceView(clip, fresh, { videoFailed: true, now })).toBe("none");
  });

  it("shows frames when there is no clip file", () => {
    expect(evidenceView({ frames: ["x"] }, old, { now })).toBe("frames");
  });

  it("trusts the API's pending flag over the event's age", () => {
    expect(evidenceView({ pending: true }, old, { now })).toBe("pending");
    expect(evidenceView({ pending: false }, fresh, { now })).toBe("none");
  });

  it("treats a fresh alert with nothing on disk as still being written", () => {
    expect(evidenceView({}, fresh, { now })).toBe("pending");
    expect(evidenceView(null, fresh, { now })).toBe("pending");
  });

  it("treats an old alert with nothing on disk as genuinely empty", () => {
    expect(evidenceView({}, old, { now })).toBe("none");
  });
});

describe("playbackProblem", () => {
  it("says nothing while the video plays", () => {
    expect(playbackProblem({ uri: "d", codec: "mp4v" }, false)).toBeNull();
    expect(playbackProblem({ frames: ["x"] }, true)).toBeNull();
  });

  it("names the codec and the fallback when it fails", () => {
    expect(playbackProblem({ uri: "d", codec: "mp4v", frames: ["x"] }, true)).toBe(
      "The clip was saved as mp4v, which this device cannot play. Showing the recorded frames instead.",
    );
  });

  it("does not blame the codec when it is already h264", () => {
    expect(playbackProblem({ uri: "d", codec: "h264", frames: [] }, true)).toBe(
      "The clip could not be played on this device. No frames were recorded alongside it.",
    );
  });
});
