import { describe, expect, it } from "vitest";
import { bridgeCameraId } from "../electron/bridge-transport.js";

describe("bridge custom protocol boundary", () => {
  it("decodes only camera requests on the registered bridge scheme", () => {
    expect(bridgeCameraId("argus-stream://camera/Front%20Door")).toBe(
      "Front Door",
    );
    expect(() => bridgeCameraId("http://camera/Front%20Door")).toThrow(
      "Invalid bridge stream request",
    );
    expect(() => bridgeCameraId("argus-stream://other/Front%20Door")).toThrow(
      "Invalid bridge stream request",
    );
    expect(() => bridgeCameraId("argus-stream://camera/")).toThrow(
      "Invalid bridge stream request",
    );
  });
});
