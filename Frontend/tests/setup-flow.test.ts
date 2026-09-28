import { describe, expect, it } from "vitest";
import {
  SETUP_STEPS,
  finishBlockedReason,
  modelReady,
  stepIndex,
} from "../src/lib/setup-flow";

describe("setup flow", () => {
  it("puts the AI model first and Finish last, with no skip step", () => {
    expect(SETUP_STEPS[0]).toBe("AI model");
    expect(SETUP_STEPS[SETUP_STEPS.length - 1]).toBe("Finish");
    expect(stepIndex("Cameras")).toBeGreaterThan(stepIndex("AI model"));
    expect(SETUP_STEPS.some((s) => /skip/i.test(s))).toBe(false);
  });
  it("only a live gate counts as a ready model", () => {
    expect(modelReady({ mode: "live" })).toBe(true);
    expect(modelReady({ mode: "no-model" })).toBe(false);
    expect(modelReady(null)).toBe(false);
  });
  it("Finish waits for the model in engine mode, and says how far along it is", () => {
    const base = { mode: "engine" as const, cameras: 2 };
    expect(finishBlockedReason({ ...base, gate: { mode: "live" }, pull: null })).toBeNull();
    expect(
      finishBlockedReason({
        ...base,
        gate: { mode: "no-model" },
        pull: { state: "pulling", percent: 43.4 },
      }),
    ).toMatch(/43%/);
    expect(
      finishBlockedReason({ ...base, gate: { mode: "no-model" }, pull: null }),
    ).toMatch(/not ready/);
  });
  it("names the offline case before a stale error", () => {
    expect(
      finishBlockedReason({
        mode: "engine",
        cameras: 1,
        gate: { mode: "no-model" },
        pull: { state: "error", detail: "connection reset" },
        online: false,
      }),
    ).toMatch(/offline/);
    expect(
      finishBlockedReason({
        mode: "engine",
        cameras: 1,
        gate: { mode: "no-model" },
        pull: { state: "error", detail: "connection reset" },
        online: true,
      }),
    ).toMatch(/connection reset/);
  });
  it("demo mode has no model to wait for, but still needs a camera", () => {
    expect(finishBlockedReason({ mode: "demo", cameras: 0, gate: null, pull: null })).toMatch(
      /camera/,
    );
    expect(finishBlockedReason({ mode: "demo", cameras: 1, gate: null, pull: null })).toBeNull();
  });
});
