import { describe, expect, it } from "vitest";
import { relativePoint, validPolygon } from "../src/lib/geometry";
describe("zone geometry", () => {
  it("scales display points to original camera pixels", () =>
    expect(
      relativePoint(
        260,
        145,
        { left: 10, top: 20, width: 500, height: 250 },
        1920,
        960,
      ),
    ).toEqual([960, 480]));
  it("clamps to image bounds", () =>
    expect(
      relativePoint(
        800,
        -2,
        { left: 0, top: 0, width: 500, height: 250 },
        1920,
        960,
      ),
    ).toEqual([1919, 0]));
  it("accepts a rectangle", () =>
    expect(
      validPolygon([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
      ]),
    ).toBe(true));
  it("rejects intersecting edges", () =>
    expect(
      validPolygon([
        [0, 0],
        [100, 100],
        [100, 0],
        [0, 80],
      ]),
    ).toBe(false));
  it("rejects empty and collinear zones", () => {
    expect(validPolygon([])).toBe(false);
    expect(
      validPolygon([
        [0, 0],
        [10, 0],
        [20, 0],
      ]),
    ).toBe(false);
  });
});
