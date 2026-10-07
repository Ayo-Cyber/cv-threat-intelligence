import { describe, expect, it } from "vitest";
import { applyPushEvent } from "../src/lib/push";
import type { Workspace } from "../src/lib/types";

const workspace: Workspace = {
  cameras: [{ id: "front", source: "rtsp://***@camera", state: "unknown" }],
  events: [
    {
      id: "evt_1",
      camera_id: "front",
      rule: "door",
      priority: "high",
      ts: 1,
      reason: "old",
      review: "new",
    },
  ],
  areas: [],
  site: {},
  monitor: { running: false },
  english: {},
  auth: {
    configured: true,
    signed_in: true,
    username: "ayo",
    role: "owner",
    permissions: [],
  },
};

describe("push state reduction", () => {
  it("retains authoritative stopped state and clears starting after warmup", () => {
    const starting = applyPushEvent(workspace, {
      type: "health", data: { engine: { phase: "starting", running: true, starting: true } },
    });
    expect(starting.monitor).toMatchObject({ running: true, starting: true });
    const ready = applyPushEvent(starting, {
      type: "health", data: { engine: { phase: "monitoring", running: true, starting: false } },
    });
    expect(ready.monitor).toMatchObject({ running: true, starting: false });
    const stopped = applyPushEvent(ready, {
      type: "health", data: { engine: { phase: "monitoring", running: false, starting: false } },
    });
    expect(stopped.monitor.running).toBe(false);
  });
  it("merges new alerts by ID without duplicating replayed events", () => {
    const event = {
      type: "alert.new" as const,
      data: {
        id: "evt_2",
        camera_id: "front",
        rule: "till",
        priority: "critical",
        ts: 2,
        reason: "new",
        review: "new",
      },
    };
    const once = applyPushEvent(workspace, event);
    const replayed = applyPushEvent(once, event);

    expect(replayed.events.map((item) => item.id)).toEqual(["evt_2", "evt_1"]);
  });

  it("replaces matching alerts and applies engine and camera health", () => {
    const updated = applyPushEvent(workspace, {
      type: "alert.update",
      data: {
        ...workspace.events[0],
        reason: "settled",
        review: "ack",
        triage_state: "acknowledged",
      },
    });
    const healthy = applyPushEvent(updated, {
      type: "health",
      data: {
        status: "ok",
        generated_at: 10,
        engine: { phase: "monitoring" },
        cameras: [
          { camera_id: "front", state: "connected", last_frame_age_s: 0.2 },
        ],
      },
    });

    expect(healthy.events[0]).toMatchObject({
      reason: "settled",
      review: "ack",
    });
    expect(healthy.monitor).toMatchObject({
      running: true,
      phase: "monitoring",
      status: "ok",
    });
    expect(healthy.cameras[0]).toMatchObject({
      state: "connected",
      last_frame_age_s: 0.2,
    });
  });
});
