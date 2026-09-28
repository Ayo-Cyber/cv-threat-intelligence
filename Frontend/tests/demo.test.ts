import { describe, it, expect } from "vitest";
import { createDemo, initialDemo } from "../src/lib/demo";
describe("isolated demo transport", () => {
  it("loads previously saved demo points as backend polygons", async () => {
    const state = initialDemo();
    const polygon = [
      [0, 0],
      [100, 0],
      [100, 100],
    ];
    const saved = {
      ...state,
      zones: {
        "Loading Bay": [
          { name: "Old demo", points: polygon, dwell_alert_seconds: 5 },
        ],
      },
    };
    const api = createDemo({
      getItem: () => JSON.stringify(saved),
      setItem: () => {},
    });
    expect(
      (await api.invoke("list_zones", ["Loading Bay"]))[0].polygon,
    ).toEqual(polygon);
  });
  it("persists detector configuration independently of engine data", async () => {
    const data = new Map<string, string>();
    const storage = {
      getItem: (k: string) => data.get(k) || null,
      setItem: (k: string, v: string) => {
        data.set(k, v);
      },
    };
    const a = createDemo(storage);
    await a.invoke("set_camera_rules", ["Loading Bay", { fire_smoke: true }]);
    const b = createDemo(storage);
    expect((await b.invoke("list_cameras"))[0].fire_smoke).toBe(true);
    expect(data.size).toBe(1);
  });
  it("never pretends to remap an image", async () => {
    await expect(
      createDemo().invoke("enqueue_scene_mapping", [["Loading Bay"]]),
    ).rejects.toThrow("requires the local engine");
  });
  it("persists zone coordinates without rescaling them", async () => {
    const a = createDemo();
    await a.invoke("add_zone", [
      "Loading Bay",
      "Reception",
      [
        [20, 20],
        [120, 20],
        [120, 120],
        [20, 120],
      ],
      5,
    ]);
    expect(
      (await a.invoke("list_zones", ["Loading Bay"]))[0].polygon[2],
    ).toEqual([120, 120]);
  });
  it("preserves a review result", async () => {
    const a = createDemo();
    await a.invoke("resolve_alert", ["sample-1", "false_alarm", "No evidence"]);
    const e = (await a.invoke("list_events"))[0];
    expect(e.review).toBe("false");
    expect(e.note).toBe("No evidence");
  });
  it("keeps unresolved separate from inconclusive resolved", async () => {
    const a = createDemo();
    await a.invoke("resolve_alert", [
      "sample-1",
      "inconclusive",
      "Cannot determine",
    ]);
    expect((await a.invoke("list_events"))[0].review).toBe("ack");
  });
  it("does not simulate evidence retention", async () => {
    await expect(createDemo().invoke("set_retention", [10])).rejects.toThrow();
  });
  it("keeps one vehicle gate line per camera and its direction", async () => {
    const a = createDemo();
    expect(
      (await a.invoke("vehicle_line", ["Loading Bay"])).vehicle_line,
    ).toBeNull();
    await a.invoke("set_vehicle_line", [
      "Loading Bay",
      [0.5, 0.05],
      [0.5, 0.95],
      true,
      "gate",
    ]);
    const line = (await a.invoke("vehicle_line", ["Loading Bay"])).vehicle_line;
    expect(line.flip).toBe(true);
    expect(line.normalized).toBe(true);
    expect(line.start).toEqual([0.5, 0.05]);
    await expect(
      a.invoke("set_vehicle_line", ["Loading Bay", [0.5, 0.5], [0.5, 0.5]]),
    ).rejects.toThrow("same point");
    await a.invoke("remove_vehicle_line", ["Loading Bay"]);
    expect(
      (await a.invoke("vehicle_line", ["Loading Bay"])).vehicle_line,
    ).toBeNull();
  });
});
