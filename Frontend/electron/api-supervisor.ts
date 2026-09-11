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
  const baseUrl = `http://127.0.0.1:${port}/api/v1`;
  let portOccupied = false;
  try {
    await fetch(baseUrl, {
      signal: AbortSignal.timeout(Math.min(1000, timeoutMs)),
    });
    portOccupied = true;
  } catch {
    /* Connection refusal or timeout means the requested port is available. */
  }
  if (portOccupied)
    throw new Error(
      `ARGUS_API_PORT port ${port} is already in use; choose a free port before starting Argus.`,
    );
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
  const started = now();
  let exit: Error | undefined;
  process.once("error", (error) => {
    exit = error;
  });
  process.once("exit", (code) => {
    exit = new Error(
      code === null
        ? "Argus API exited during startup."
        : `Argus API exited during startup (code ${code})`,
    );
  });
  while (now() - started < timeoutMs) {
    if (exit) throw exit;
    const remaining = Math.max(1, timeoutMs - (now() - started));
    let response: Response | undefined;
    try {
      response = await fetch(baseUrl, {
        signal: AbortSignal.timeout(Math.min(1000, remaining)),
      });
    } catch {
      /* The spawned server socket is not listening yet. */
    }
    if (exit || process.exitCode !== null || process.killed)
      throw exit ?? new Error("Owned Argus API process exited during startup.");
    if (response?.ok) {
      let identity: any;
      try {
        identity = await response.json();
      } catch {
        identity = undefined;
      }
      if (identity?.name !== "Argus Engine API" || identity?.status !== "ok") {
        if (process.exitCode === null && !process.killed)
          process.kill("SIGTERM");
        throw new Error(
          `Service on port ${port} is not the owned Argus Engine API. Check runs/desktop/frontend.log.`,
        );
      }
      if (exit || process.exitCode !== null || process.killed)
        throw (
          exit ?? new Error("Owned Argus API process exited during startup.")
        );
      return {
        baseUrl,
        process,
        async stop() {
          if (process.exitCode === null && !process.killed)
            process.kill("SIGTERM");
        },
      };
    }
    await sleep(250);
  }
  if (process.exitCode === null && !process.killed) process.kill("SIGTERM");
  throw new Error(
    `Argus API did not become ready within ${timeoutMs / 1000} seconds. Check runs/desktop/frontend.log.`,
  );
}
