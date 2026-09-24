import { describe, expect, it } from "vitest";
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { shouldAdHocSign } = require("../scripts/after-pack.cjs");

/**
 * 24 Sep: a fresh v1.8.22 download said "Argus is damaged and can't be
 * opened." An unsigned app on Apple Silicon is not reported as unsigned, it
 * is reported as corrupt. An ad-hoc signature costs nothing and turns that
 * into the ordinary unidentified-developer prompt.
 */
describe("ad-hoc signing decision", () => {
  it("the hook is a tracked file electron-builder can actually load", () => {
    // .gitignore's `build/` silently swallowed this file on the first
    // attempt: `git add -A` skipped it, and every OS build then failed with
    // "Cannot find module". It lives beside preload.cjs now, which is tracked.
    expect(typeof shouldAdHocSign).toBe("function");
  });

  it("signs a Mac build when no certificate is present", () => {
    expect(shouldAdHocSign("darwin", {})).toBe(true);
  });

  it("leaves a properly signed build alone", () => {
    expect(shouldAdHocSign("darwin", { CSC_LINK: "/tmp/cert.p12" })).toBe(false);
    expect(shouldAdHocSign("darwin", { MACOS_CERT_P12: "base64..." })).toBe(false);
  });

  it("never touches Windows or Linux builds", () => {
    expect(shouldAdHocSign("win32", {})).toBe(false);
    expect(shouldAdHocSign("linux", {})).toBe(false);
  });
});
