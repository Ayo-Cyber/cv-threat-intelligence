import { describe, expect, it } from "vitest";
import {
  cameraTrackingPreference,
  loadGlobalTrackingPreference,
  saveGlobalTrackingPreference,
  trackingVisible,
  updateTrackingOverride,
} from "../src/lib/tracking-overlay";

function memoryStorage(initial: Record<string, string> = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
  };
}

describe("tracking overlay preferences", () => {
  it("defaults to a clean feed when the operator has no saved preference", () => {
    expect(loadGlobalTrackingPreference(memoryStorage(), "operator-1")).toBe(
      false,
    );
  });

  it("restores a shown global preference for the same operator only", () => {
    const storage = memoryStorage();

    saveGlobalTrackingPreference(storage, "operator-1", true);

    expect(loadGlobalTrackingPreference(storage, "operator-1")).toBe(true);
    expect(loadGlobalTrackingPreference(storage, "operator-2")).toBe(false);
  });

  it("treats malformed persisted values as hidden", () => {
    const storage = memoryStorage({
      "argus:tracking-overlay:operator-1": "yes",
    });

    expect(loadGlobalTrackingPreference(storage, "operator-1")).toBe(false);
  });

  it("lets show override a hidden global preference", () => {
    expect(trackingVisible(false, "show")).toBe(true);
  });

  it("lets hide override a shown global preference", () => {
    expect(trackingVisible(true, "hide")).toBe(false);
  });

  it("drops camera overrides when a new session starts", () => {
    const previousSession = updateTrackingOverride({}, "front-door", "show");
    const nextSession = {};

    expect(cameraTrackingPreference(previousSession, "front-door")).toBe(
      "show",
    );
    expect(cameraTrackingPreference(nextSession, "front-door")).toBe("global");
  });
});
