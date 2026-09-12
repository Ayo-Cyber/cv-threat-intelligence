import { expect, test, type Page } from "@playwright/test";

const jpeg = Buffer.from(
  "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAEf/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABAf/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPxB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPxB//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxB//9k=",
  "base64",
);

async function openHarness(page: Page) {
  await page.goto("/tests/fixtures/camera-stream.html");
  await page.waitForFunction(() => Boolean((window as any).cameraHarness));
}

async function render(page: Page, options: Record<string, unknown>) {
  await page.evaluate(
    (value) => (window as any).cameraHarness.render(value),
    options,
  );
}

test("actual MJPEG onLoad gates live status and health rerenders override it", async ({
  page,
}) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => (release = resolve));
  await page.route("**/camera.mjpeg", async (route) => {
    await gate;
    await route.fulfill({ contentType: "image/jpeg", body: jpeg });
  });
  await openHarness(page);
  const descriptor = { kind: "mjpeg", url: "/camera.mjpeg" };
  await render(page, { state: "connected", descriptor });
  await expect(page.locator(".media-label")).toHaveText("CONNECTING");
  release();
  await expect(page.locator(".media-label")).toHaveText("MJPEG FEED");

  await render(page, { state: "offline", descriptor });
  await expect(page.locator(".media-label")).toHaveText("OFFLINE");
  await expect(
    page.getByText("Camera health reports that this feed is offline."),
  ).toBeVisible();

  await render(page, { state: "reconnecting", descriptor });
  await expect(page.locator(".media-label")).toHaveText("RECONNECTING");
});

test("actual WebRTC ontrack gates live status and degraded health overrides it", async ({
  page,
}) => {
  await page.route("**/whep", (route) => route.fulfill({ body: "answer" }));
  await openHarness(page);
  const descriptor = {
    kind: "webrtc",
    url: "http://127.0.0.1:5173/whep",
    mjpeg_fallback: "/fallback.mjpeg",
  };
  await render(page, { state: "connected", descriptor });
  await expect(page.locator(".media-label")).toHaveText("CONNECTING");
  await expect
    .poll(() => page.evaluate(() => (window as any).cameraHarness.emitTrack()))
    .toBe(true);
  await expect(page.locator(".media-label")).toHaveText("WEBRTC FEED");

  await render(page, { state: "degraded", descriptor });
  await expect(page.locator(".media-label")).toHaveText("DEGRADED");
});

test("actual fallback img onLoad gates its degraded label and inactive stays inactive", async ({
  page,
}) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => (release = resolve));
  await page.route("**/fallback.mjpeg", async (route) => {
    await gate;
    await route.fulfill({ contentType: "image/jpeg", body: jpeg });
  });
  await openHarness(page);
  const descriptor = {
    kind: "webrtc",
    url: "http://127.0.0.1:5173/whep",
    mjpeg_fallback: "/fallback.mjpeg",
  };
  await render(page, { state: "connected", descriptor, failWhep: true });
  await expect(page.locator(".media-label")).toHaveText("CONNECTING");
  release();
  await expect(page.locator(".media-label")).toHaveText("MJPEG FALLBACK");

  await render(page, { state: "connected", descriptor, active: false });
  await expect(page.locator(".media-label")).toHaveText("INACTIVE");
});
