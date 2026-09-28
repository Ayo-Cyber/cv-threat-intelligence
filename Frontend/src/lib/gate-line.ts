import type { Point, VehicleLine } from "./types";

/** The vehicle tripwire (KPI 2: vehicle entering / exiting). The engine
 * builds a supervision LineZone from `start` -> `end` and counts a
 * vehicle whose centre moves onto the LEFT of that vector (walking from
 * start to end, y pointing down) as ENTERING. `flip` swaps the sides.
 * Everything here is the pure geometry the editor draws and sends, so the
 * arrow on screen and the direction the engine fires on cannot drift. */

/** A line needs two distinct points that are both inside the frame. */
export function validLine(a: Point | null, b: Point | null): boolean {
  if (!a || !b) return false;
  if ([...a, ...b].some((v) => !Number.isFinite(v))) return false;
  return Math.hypot(b[0] - a[0], b[1] - a[1]) >= 8;
}

/** Original-frame pixels -> fractions of the frame (what set_vehicle_line
 * stores as `normalized: true`, so the line survives a resolution change). */
export function toFraction(p: Point, size: { width: number; height: number }): Point {
  const w = Math.max(1, size.width);
  const h = Math.max(1, size.height);
  return [
    Math.round(Math.min(1, Math.max(0, p[0] / w)) * 10000) / 10000,
    Math.round(Math.min(1, Math.max(0, p[1] / h)) * 10000) / 10000,
  ];
}

/** A saved line (fractions or pixels — the record says which) -> pixels of
 * the frame the editor is showing. */
export function toPixels(
  line: VehicleLine,
  size: { width: number; height: number },
): { start: Point; end: Point } {
  const norm =
    line.normalized ??
    Math.max(line.start[0], line.start[1], line.end[0], line.end[1]) <= 1;
  const conv = (p: [number, number]): Point =>
    norm ? [p[0] * size.width, p[1] * size.height] : [p[0], p[1]];
  return { start: conv(line.start), end: conv(line.end) };
}

/** Unit vector pointing from the line to its ENTER side. supervision's
 * cross product `(dx*py - dy*px)` is positive on the left of start->end;
 * the normal (-dy, dx) lands there. `flip` sends it the other way. */
export function enterNormal(start: Point, end: Point, flip = false): Point {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const len = Math.hypot(dx, dy) || 1;
  const s = flip ? -1 : 1;
  return [(-dy / len) * s, (dx / len) * s];
}

/** The arrow drawn across the middle of the line: from the EXIT side,
 * through the line, to the ENTER side. `reach` is in frame pixels. */
export function enterArrow(
  start: Point,
  end: Point,
  flip = false,
  reach = 40,
): { from: Point; to: Point } {
  const mid: Point = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2];
  const n = enterNormal(start, end, flip);
  return {
    from: [mid[0] - n[0] * reach, mid[1] - n[1] * reach],
    to: [mid[0] + n[0] * reach, mid[1] + n[1] * reach],
  };
}

/** Which side of the frame vehicles ENTER from, in words an operator can
 * check against the arrow: "vehicles entering move left → right". */
export function enterDirectionLabel(start: Point, end: Point, flip = false): string {
  const [nx, ny] = enterNormal(start, end, flip);
  if (Math.abs(nx) >= Math.abs(ny))
    return nx > 0 ? "left → right" : "right → left";
  return ny > 0 ? "top → bottom" : "bottom → top";
}
