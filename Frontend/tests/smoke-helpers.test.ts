import { describe, expect, it } from "vitest";
import { fullscreenSkipAllowed } from "../scripts/smoke_api_transport.mjs";

describe("native fullscreen smoke capability", () => {
  it("allows a skip only when requestFullscreen is absent", () => {
    expect(
      fullscreenSkipAllowed({
        requestFullscreen: false,
        fullscreenEnabled: false,
      }),
    ).toBe(true);
    expect(
      fullscreenSkipAllowed({
        requestFullscreen: true,
        fullscreenEnabled: false,
      }),
    ).toBe(false);
    expect(
      fullscreenSkipAllowed({
        requestFullscreen: true,
        fullscreenEnabled: true,
      }),
    ).toBe(false);
  });
});
