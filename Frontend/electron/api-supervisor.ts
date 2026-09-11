import {
  spawn as nodeSpawn,
  type ChildProcessWithoutNullStreams,
} from "node:child_process";

type StartOptions = {
  python: string;
  root: string;
  site: string;
  db: string;
  port: number;
  spawn?: typeof nodeSpawn;
  fetch?: typeof globalThis.fetch;
  sleep?: (milliseconds: number) => Promise<void>;
  now?: () => number;
  timeoutMs?: number;
  onStderr?: (data: Buffer) => void;
};

export type OwnedApi = {
  baseUrl: string;
  process: ChildProcessWithoutNullStreams;
  stop(): Promise<void>;
};

export async function startOwnedApi({
  python,
  root,
  site,
  db,
  port,
  spawn = nodeSpawn,
  fetch = globalThis.fetch,
  sleep = (milliseconds) =>
    new Promise((resolve) => setTimeout(resolve, milliseconds)),
  now = Date.now,
  timeoutMs = 30_000,
  onStderr,
}: StartOptions): Promise<OwnedApi> {
  const process = spawn(
    python,
    [
      "-u",
      "-m",
      "cvti.api",
      "--host",
      "127.0.0.1",
      "--port",
      String(port),
      "--site",
      site,
      "--db",
      db,
    ],
    {
      cwd: root,
      env: { ...globalThis.process.env, PYTHONUNBUFFERED: "1" },
      stdio: "pipe",
    },
  );
  if (onStderr) process.stderr.on("data", onStderr);
  const baseUrl = `http://127.0.0.1:${port}/api/v1`;
  const started = now();
  let exit: Error | undefined;
  process.once("error", (error) => {
    exit = error;
  });
  process.once("exit", (code) => {
    if (code !== null && code !== 0)
      exit = new Error(`Argus API exited during startup (code ${code})`);
  });
  while (now() - started < timeoutMs) {
    if (exit) throw exit;
    try {
      const remaining = Math.max(1, timeoutMs - (now() - started));
      const response = await fetch(baseUrl, {
        signal: AbortSignal.timeout(Math.min(1000, remaining)),
      });
      if (response.ok)
        return {
          baseUrl,
          process,
          async stop() {
            if (process.exitCode === null && !process.killed)
              process.kill("SIGTERM");
          },
        };
    } catch {
      // The server socket is not listening yet.
    }
    await sleep(250);
  }
  if (process.exitCode === null && !process.killed) process.kill("SIGTERM");
  throw new Error(
    `Argus API did not become ready within ${timeoutMs / 1000} seconds. Check runs/desktop/frontend.log.`,
  );
}
