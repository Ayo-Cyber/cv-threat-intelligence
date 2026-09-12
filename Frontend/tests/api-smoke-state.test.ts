import { createHash } from "node:crypto";
import os from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { createSmokeStateWriter } from "../electron/api-runtime.js";

describe("API smoke state", () => {
  const root = path.resolve(os.tmpdir(), "argus-smoke-state-test");
  const profile = path.join(root, "profile");
  const state = path.join(profile, "api-state.json");
  const outside = path.join(root, "outside.json");

  it("is disabled unless the explicit smoke gate is enabled", () => {
    expect(
      createSmokeStateWriter({
        ARGUS_USER_DATA: profile,
        ARGUS_SMOKE_STATE: state,
      }),
    ).toBeUndefined();
  });

  it("rejects a smoke state path outside the temporary user profile", () => {
    expect(() =>
      createSmokeStateWriter({
        ARGUS_SMOKE_TEST: "1",
        ARGUS_USER_DATA: profile,
        ARGUS_SMOKE_STATE: outside,
      }),
    ).toThrow("must be inside ARGUS_USER_DATA");
  });

  it("writes only the owned PID and token fingerprint", () => {
    const mkdirSync = vi.fn();
    const writeFileSync = vi.fn();
    const writer = createSmokeStateWriter(
      {
        ARGUS_SMOKE_TEST: "1",
        ARGUS_USER_DATA: profile,
        ARGUS_SMOKE_STATE: state,
      },
      { mkdirSync, writeFileSync },
    );

    writer?.recordPid(4321);
    writer?.recordToken("random-main-token");

    expect(mkdirSync).toHaveBeenCalledWith(profile, { recursive: true });
    const [, body, options] = writeFileSync.mock.calls.at(-1) || [];
    expect(JSON.parse(String(body))).toEqual({
      api_pid: 4321,
      token_sha256: createHash("sha256")
        .update("random-main-token")
        .digest("hex"),
    });
    expect(String(body)).not.toContain("random-main-token");
    expect(options).toEqual({ encoding: "utf8", mode: 0o600 });
  });
});
