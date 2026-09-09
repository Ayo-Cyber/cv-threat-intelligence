import type { Point } from "./types";
export function relativePoint(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  width: number,
  height: number,
): Point {
  return [
    Math.round(
      Math.max(
        0,
        Math.min(width - 1, ((clientX - rect.left) / rect.width) * width),
      ),
    ),
    Math.round(
      Math.max(
        0,
        Math.min(height - 1, ((clientY - rect.top) / rect.height) * height),
      ),
    ),
  ];
}
export function validPolygon(points: Point[]): boolean {
  if (points.some((point) => point.some((value) => !Number.isFinite(value))))
    return false;
  if (points.length < 3 || new Set(points.map((p) => p.join(","))).size < 3)
    return false;
  const area =
    Math.abs(
      points.reduce((sum, p, i) => {
        const next = points[(i + 1) % points.length];
        return sum + p[0] * next[1] - next[0] * p[1];
      }, 0),
    ) / 2;
  if (area < 25) return false;
  const cross = (a: Point, b: Point, c: Point) =>
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  for (let i = 0; i < points.length; i++)
    for (let j = i + 2; j < points.length; j++) {
      if (i === 0 && j === points.length - 1) continue;
      const a = points[i],
        b = points[(i + 1) % points.length],
        c = points[j],
        d = points[(j + 1) % points.length];
      if (
        cross(a, b, c) * cross(a, b, d) <= 0 &&
        cross(c, d, a) * cross(c, d, b) <= 0
      )
        return false;
    }
  return true;
}
