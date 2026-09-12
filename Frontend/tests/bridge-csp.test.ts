import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

describe("bridge stream CSP", () => {
  it("allows the Electron bridge scheme as an image source", () => {
    const html = fs.readFileSync(path.resolve("index.html"), "utf8");
    const policy = html.match(
      /http-equiv="Content-Security-Policy"\s+content="([^"]+)"/,
    )?.[1];
    const imageSources = policy
      ?.split(";")
      .map((directive) => directive.trim().split(/\s+/))
      .find(([name]) => name === "img-src")
      ?.slice(1);

    expect(imageSources).toContain("argus-stream:");
  });
});
