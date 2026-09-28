import { describe, expect, it } from "vitest";
import {
  enterArrow,
  enterDirectionLabel,
  enterNormal,
  toFraction,
  toPixels,
  validLine,
} from "../src/lib/gate-line";

/** supervision's LineZone counts a centre that lands on the LEFT of the
 * start->end vector (y down) as "in". The engine turns that into
 * vehicle_entry, with `flip` swapping the sides. The arrow the editor draws
 * has to point at exactly that side, or the operator draws the line, sees
 * an arrow, and gets EXIT alerts for vehicles driving in. */
describe("gate line direction", () => {
  it("a top-to-bottom line enters from the right, moving left", () => {
    // cross = dx*py - dy*px with dx = 0, dy > 0  ->  positive when px < 0
    const n = enterNormal([100, 0], [100, 200]);
    expect(n[0]).toBeCloseTo(-1);
    expect(n[1]).toBeCloseTo(0);
    expect(enterDirectionLabel([100, 0], [100, 200])).toBe("right → left");
  });
  it("flip swaps the entering side", () => {
    const n = enterNormal([100, 0], [100, 200], true);
    expect(n[0]).toBeCloseTo(1);
    expect(enterDirectionLabel([100, 0], [100, 200], true)).toBe(
      "left → right",
    );
  });
  it("a left-to-right line enters from above, moving down", () => {
    // dx > 0, dy = 0 -> cross = dx*py positive when py > 0 (below, y down)
    expect(enterDirectionLabel([0, 100], [200, 100])).toBe("top → bottom");
    expect(enterDirectionLabel([200, 100], [0, 100])).toBe("bottom → top");
  });
  it("the arrow crosses the middle of the line towards the enter side", () => {
    const a = enterArrow([100, 0], [100, 200], false, 40);
    expect(a.from).toEqual([140, 100]);
    expect(a.to).toEqual([60, 100]);
  });
});

describe("gate line coordinates", () => {
  it("sends fractions of the frame, rounded, never outside 0..1", () => {
    const size = { width: 1280, height: 720 };
    expect(toFraction([640, 360], size)).toEqual([0.5, 0.5]);
    expect(toFraction([1279, 719], size)).toEqual([0.9992, 0.9986]);
    expect(toFraction([-5, 9999], size)).toEqual([0, 1]);
  });
  it("draws a saved normalized line at the frame being shown", () => {
    const px = toPixels(
      { name: "gate", start: [0.42, 0.05], end: [0.42, 0.95], normalized: true },
      { width: 1000, height: 500 },
    );
    expect(px.start).toEqual([420, 25]);
    expect(px.end).toEqual([420, 475]);
  });
  it("leaves a pixel line alone", () => {
    const px = toPixels(
      { name: "gate", start: [300, 10], end: [300, 400], normalized: false },
      { width: 1000, height: 500 },
    );
    expect(px.start).toEqual([300, 10]);
  });
  it("infers normalization from the record when it is missing", () => {
    const px = toPixels(
      { name: "gate", start: [0.5, 0.1], end: [0.5, 0.9] },
      { width: 200, height: 100 },
    );
    expect(px.end).toEqual([100, 90]);
  });
  it("needs two distinct points", () => {
    expect(validLine(null, [1, 1])).toBe(false);
    expect(validLine([10, 10], [12, 11])).toBe(false);
    expect(validLine([10, 10], [200, 10])).toBe(true);
    expect(validLine([NaN, 10], [200, 10])).toBe(false);
  });
});
