import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { createSmokeStateWriter } from "../electron/api-runtime.js";

describe("API smoke state", () => {
  it("is disabled unless the explicit smoke gate is enabled", () => {
    expect(
      createSmokeStateWriter({
        ARGUS_USER_DATA: "/tmp/profile",
        ARGUS_SMOKE_STATE: "/tmp/profile/state.json",
      }),
    ).toBeUndefined();
  });

  it("rejects a smoke state path outside the temporary user profile", () => {
    expect(() =>
      createSmokeStateWriter({
        ARGUS_SMOKE_TEST: "1",
        ARGUS_USER_DATA: "/tmp/profile",
        ARGUS_SMOKE_STATE: "/tmp/outside.json",
      }),
    ).toThrow("must be inside ARGUS_USER_DATA");
  });

  it("writes only the owned PID and token fingerprint", () => {
    const mkdirSync = vi.fn();
    const writeFileSync = vi.fn();
    const writer = createSmokeStateWriter(
      {
        ARGUS_SMOKE_TEST: "1",
        ARGUS_USER_DATA: "/tmp/profile",
        ARGUS_SMOKE_STATE: "/tmp/profile/api-state.json",
      },
      { mkdirSync, writeFileSync },
    );

    writer?.recordPid(4321);
    writer?.recordToken("random-main-token");

    expect(mkdirSync).toHaveBeenCalledWith("/tmp/profile", { recursive: true });
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
