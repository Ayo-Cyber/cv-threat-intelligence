import fs from "node:fs";
import path from "node:path";

const DEFAULT_MAX_BYTES = 2 * 1024 * 1024;

export function redactSupportLog(value: string | Buffer): string {
  return String(value)
    .replace(/(\bBearer\s+)[A-Za-z0-9._~+/-]+/gi, "$1[REDACTED]")
    .replace(/(argus\.token\.)[A-Za-z0-9_-]+/gi, "$1[REDACTED]")
    .replace(/([?&]\btoken=)[^&\s]+/gi, "$1[REDACTED]")
    .replace(/(\btoken\s*=\s*)[^\s&,;]+/gi, "$1[REDACTED]")
    .replace(/("token"\s*:\s*")[^"]+/gi, "$1[REDACTED]");
}

export function createSupportLog(
  logPath: string,
  { maxBytes = DEFAULT_MAX_BYTES }: { maxBytes?: number } = {},
) {
  const directory = path.dirname(logPath);
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  fs.chmodSync(directory, 0o700);
  if (!fs.existsSync(logPath)) fs.writeFileSync(logPath, "", { mode: 0o600 });
  fs.chmodSync(logPath, 0o600);

  let pending = "";
  let closed = false;

  function append(safe: string): string {
    if (!safe) return "";
    const incoming = Buffer.from(safe);
    if (incoming.length >= maxBytes) {
      fs.writeFileSync(logPath, incoming.subarray(incoming.length - maxBytes), {
        mode: 0o600,
      });
    } else {
      const currentSize = fs.statSync(logPath).size;
      if (currentSize + incoming.length > maxBytes) {
        const retainedBytes = maxBytes - incoming.length;
        const current = fs.readFileSync(logPath);
        fs.writeFileSync(
          logPath,
          Buffer.concat([
            current.subarray(Math.max(0, current.length - retainedBytes)),
            incoming,
          ]),
          { mode: 0o600 },
        );
      } else {
        fs.appendFileSync(logPath, incoming, { mode: 0o600 });
      }
    }
    fs.chmodSync(logPath, 0o600);
    return safe;
  }

  function flush(): string {
    if (!pending) return "";
    const safe = redactSupportLog(pending);
    pending = "";
    return append(safe);
  }

  return {
    write(data: string | Buffer): string {
      if (closed) return "";
      pending += String(data);
      const boundary = pending.lastIndexOf("\n");
      if (boundary < 0) return "";
      const complete = pending.slice(0, boundary + 1);
      pending = pending.slice(boundary + 1);
      return append(redactSupportLog(complete));
    },
    flush,
    close(): string {
      if (closed) return "";
      const safe = flush();
      closed = true;
      return safe;
    },
  };
}
