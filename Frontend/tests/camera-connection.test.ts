import { describe, expect, it } from "vitest";
import { VENDORS, buildRtspUrl } from "../src/components/CameraConnection";

/**
 * The guided form the React shell dropped. Its whole job is that an installer
 * should not have to know a vendor's stream path, and that a camera password
 * cannot corrupt the URL it goes into.
 */
describe("building a camera address", () => {
  it("assembles the address the pilot's Tapo actually needs", () => {
    expect(
      buildRtspUrl("192.168.8.100", "/stream1", "okwesimartins", "12345678"),
    ).toBe("rtsp://okwesimartins:12345678@192.168.8.100:554/stream1");
  });

  it("percent-encodes credentials so punctuation cannot break the URL", () => {
    // A password with @ or : would otherwise end the userinfo early and point
    // the engine at a different host entirely.
    const url = buildRtspUrl("10.0.2.5", "/stream1", "admin", "p@ss:w/rd");
    expect(url).toBe("rtsp://admin:p%40ss%3Aw%2Frd@10.0.2.5:554/stream1");
    expect(new URL(url).hostname).toBe("10.0.2.5");
  });

  it("omits the credential section entirely when there is no username", () => {
    expect(buildRtspUrl("10.0.2.5", "/stream1")).toBe("rtsp://10.0.2.5:554/stream1");
  });

  it("keeps a camera off the default port when one is given", () => {
    expect(buildRtspUrl("10.0.2.5", "/stream1", "", "", "8554")).toBe(
      "rtsp://10.0.2.5:8554/stream1",
    );
  });

  it("returns nothing until an address is typed", () => {
    expect(buildRtspUrl("   ", "/stream1", "admin", "pw")).toBe("");
  });

  it("carries the vendor paths the old console knew", () => {
    const paths = Object.fromEntries(VENDORS.map((v) => [v.label, v.path]));
    expect(paths["Hikvision"]).toBe("/Streaming/Channels/102");
    expect(paths["Dahua / Amcrest"]).toBe("/cam/realmonitor?channel=1&subtype=1");
    expect(paths["Reolink"]).toBe("/h264Preview_01_sub");
    // Tapo is the pilot's camera and was only reachable as "Generic" before.
    expect(paths["Tapo / TP-Link"]).toBe("/stream2");
  });
});
