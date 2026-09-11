const http = require("node:http");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { app, BrowserWindow, net, protocol } = require("electron");

protocol.registerSchemesAsPrivileged([
  {
    scheme: "argus-stream",
    privileges: { standard: true, secure: true, stream: true },
  },
]);

const jpeg = Buffer.from(
  "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAEf/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABAf/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPxB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPxB//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxB//9k=",
  "base64",
);
let server;

app.whenReady().then(async () => {
  server = http.createServer((request, response) => {
    if (!request.url?.startsWith("/stream/Front%20Door?token=fixture-token")) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      "content-type": "multipart/x-mixed-replace; boundary=frame",
      "cache-control": "no-store",
    });
    response.end(
      Buffer.concat([
        Buffer.from(
          `--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.length}\r\n\r\n`,
        ),
        jpeg,
        Buffer.from("\r\n--frame--\r\n"),
      ]),
    );
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  const moduleUrl = pathToFileURL(
    path.resolve(__dirname, "../../dist-electron/bridge-transport.js"),
  ).href;
  const { createBridgeTransport, registerBridgeStreamProtocol } = await import(
    moduleUrl
  );
  const bridge = createBridgeTransport(
    async (method) => {
      if (method === "live_start") return { port, token: "fixture-token" };
      throw new Error(`Unexpected bridge operation: ${method}`);
    },
    (url) => net.fetch(url),
  );
  await registerBridgeStreamProtocol(protocol, bridge);
  const descriptor = await bridge.invoke("camera_stream", ["Front Door"]);
  const window = new BrowserWindow({
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  await window.loadFile(path.resolve(__dirname, "../../dist/index.html"));
  await window.webContents.executeJavaScript(
    `window.__bridgeDescriptor = ${JSON.stringify(descriptor.url)}`,
  );
});

app.on("before-quit", () => server?.close());
app.on("window-all-closed", () => app.quit());
