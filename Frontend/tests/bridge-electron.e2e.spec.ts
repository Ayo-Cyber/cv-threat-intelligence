import path from "node:path";
import { _electron as electron, expect, test } from "@playwright/test";

test("production CSP loads bridge MJPEG through the registered protocol", async () => {
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  const errors: string[] = [];
  const requests: string[] = [];
  const failures: string[] = [];
  const electronApp = await electron.launch({
    args: [path.resolve("tests/fixtures/bridge-electron-main.cjs")],
    env,
  });
  try {
    const window = await electronApp.firstWindow();
    await window.waitForFunction(
      () => typeof (window as any).__bridgeDescriptor === "string",
    );
    window.on("request", (request) => requests.push(request.url()));
    window.on("requestfailed", (request) => {
      if (request.url().startsWith("argus-stream:"))
        failures.push(`${request.url()}: ${request.failure()?.errorText}`);
    });
    window.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    const loaded = await window.evaluate(
      () =>
        new Promise<{
          src: string;
          width: number;
          height: number;
          error?: boolean;
        }>((resolve) => {
          const image = document.createElement("img");
          image.id = "bridge-preview";
          image.onload = () =>
            resolve({
              src: image.currentSrc,
              width: image.naturalWidth,
              height: image.naturalHeight,
            });
          image.onerror = () =>
            resolve({
              src: image.currentSrc || image.src,
              width: image.naturalWidth,
              height: image.naturalHeight,
              error: true,
            });
          image.src = (window as any).__bridgeDescriptor;
          document.body.append(image);
        }),
    );

    expect({ loaded, requests, failures, errors }).toEqual({
      loaded: {
        src: "argus-stream://camera/Front%20Door",
        width: 1,
        height: 1,
      },
      requests: expect.arrayContaining(["argus-stream://camera/Front%20Door"]),
      failures: [],
      errors: [],
    });
  } finally {
    await electronApp.close();
  }
});
