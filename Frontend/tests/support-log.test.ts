import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createSupportLog } from "../electron/support-log.js";

const directories: string[] = [];

describe("desktop support log", () => {
  afterEach(() => {
    for (const directory of directories.splice(0))
      fs.rmSync(directory, { recursive: true, force: true });
  });

  it("redacts credentials and creates restrictive support files", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "argus-log-"));
    directories.push(directory);
    const logPath = path.join(directory, "private", "frontend.log");
    const log = createSupportLog(logPath, { maxBytes: 4096 });
    const token = "raw-main-process-token-123456789";

    log.write(
      `Authorization: Bearer ${token}\nGET /stream?token=${token}\nSec-WebSocket-Protocol: argus.v1,argus.token.${token}\n`,
    );

    const contents = fs.readFileSync(logPath, "utf8");
    expect(contents).not.toContain(token);
    expect(contents).toContain("[REDACTED]");
    expect(fs.statSync(logPath).mode & 0o777).toBe(0o600);
    expect(fs.statSync(path.dirname(logPath)).mode & 0o777).toBe(0o700);
  });

  it("bounds retained output without reintroducing redacted text", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "argus-log-"));
    directories.push(directory);
    const logPath = path.join(directory, "frontend.log");
    const log = createSupportLog(logPath, { maxBytes: 256 });

    for (let index = 0; index < 20; index += 1)
      log.write(`line-${index} token=secret-${index}-${"x".repeat(32)}\n`);

    expect(fs.statSync(logPath).size).toBeLessThanOrEqual(256);
    expect(fs.readFileSync(logPath, "utf8")).not.toMatch(/secret-\d+/);
  });

  it.each([
    ["bearer header", "Authorization: Bearer split-secret-123"],
    ["query token", "GET /stream?token=split-secret-123"],
    [
      "websocket protocol token",
      "Sec-WebSocket-Protocol: argus.v1,argus.token.split-secret-123",
    ],
    ["token assignment", "status token=split-secret-123"],
    ["JSON token", '{"token":"split-secret-123"}'],
  ])("redacts a %s split at every chunk boundary", (_label, credential) => {
    for (let split = 1; split < credential.length; split += 1) {
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), "argus-log-"));
      directories.push(directory);
      const logPath = path.join(directory, "frontend.log");
      const log = createSupportLog(logPath, { maxBytes: 4096 });

      const emitted =
        log.write(credential.slice(0, split)) +
        log.write(`${credential.slice(split)}\n`) +
        log.close();
      const contents = fs.readFileSync(logPath, "utf8");

      expect(emitted, `emitted split ${split}`).not.toContain(
        "split-secret-123",
      );
      expect(contents, `stored split ${split}`).not.toContain(
        "split-secret-123",
      );
      expect(contents, `redacted split ${split}`).toContain("[REDACTED]");
    }
  });

  it("redacts an unterminated credential when the log is flushed or closed", () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "argus-log-"));
    directories.push(directory);
    const flushedPath = path.join(directory, "flushed.log");
    const flushed = createSupportLog(flushedPath, { maxBytes: 4096 });

    expect(flushed.write("Authorization: Bear")).toBe("");
    expect(flushed.write("er flush-secret-123")).toBe("");
    expect(flushed.flush()).not.toContain("flush-secret-123");
    expect(fs.readFileSync(flushedPath, "utf8")).not.toContain(
      "flush-secret-123",
    );

    const closedPath = path.join(directory, "closed.log");
    const closed = createSupportLog(closedPath, { maxBytes: 4096 });
    closed.write("token=close-");
    closed.write("secret-123");
    expect(closed.close()).not.toContain("close-secret-123");
    expect(fs.readFileSync(closedPath, "utf8")).not.toContain(
      "close-secret-123",
    );
  });
});
