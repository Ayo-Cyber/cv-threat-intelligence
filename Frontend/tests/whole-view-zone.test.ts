import { describe, expect, it } from "vitest";
import { wholeViewPolygon } from "../src/components/ZoneEditor";

describe("one-click whole-view zone", () => {
  it("covers the camera's own frame in original pixels", () => {
    expect(wholeViewPolygon({ width: 1920, height: 1080 })).toEqual([
      [0, 0], [1920, 0], [1920, 1080], [0, 1080],
    ]);
  });
  it("rounds and never collapses to a degenerate polygon", () => {
    expect(wholeViewPolygon({ width: 639.6, height: 0 })).toEqual([
      [0, 0], [640, 0], [640, 1], [0, 1],
    ]);
  });
});
