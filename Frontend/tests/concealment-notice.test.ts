import { describe, expect, it } from "vitest";
import { activeConcealmentNotice } from "../src/lib/concealment-notice";

describe("concealment notices", () => {
  const notice = { id: "a", phase: "verifying", created_at: 100, expires_at: 108 };
  it("expires by engine timestamp, not repeated heartbeat receipt", () => {
    expect(activeConcealmentNotice(notice, 101000)).toEqual(notice);
    expect(activeConcealmentNotice(notice, 108000)).toBeNull();
  });
  it("rejects stale or malformed notices", () => {
    for (const value of [null, {}, { ...notice, phase: "confirmed-theft" },
      { ...notice, expires_at: Infinity }, { ...notice, expires_at: 150 }]) {
      expect(activeConcealmentNotice(value, 101000)).toBeNull();
    }
  });
});
