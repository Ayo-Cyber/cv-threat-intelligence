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
  fetch: typeof globalThis.fetch;
  assertAlive(): void;
  onExit(listener: (error: Error) => void): () => void;
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

  const lifecycle = new AbortController();
  const exitListeners = new Set<(error: Error) => void>();
  let exitError: Error | undefined;
  const markExited = (error: Error) => {
    if (exitError) return;
    exitError = error;
    lifecycle.abort(error);
    for (const listener of exitListeners) listener(error);
    exitListeners.clear();
  };
  process.once("error", markExited);
  process.once("exit", (code) =>
    markExited(
      new Error(
        code === null
          ? "Argus API exited."
          : `Argus API exited (code ${code}).`,
      ),
    ),
  );

  const assertAlive = () => {
    if (exitError) throw exitError;
    if (process.exitCode !== null || process.killed) {
      markExited(new Error("Owned Argus API process is not running."));
      throw exitError;
    }
  };
  const guardedFetch: typeof globalThis.fetch = async (input, init) => {
    assertAlive();
    const signal = init?.signal
      ? AbortSignal.any([init.signal, lifecycle.signal])
      : lifecycle.signal;
    try {
      const response = await fetch(input, { ...init, signal });
      assertAlive();
      return response;
    } catch (error) {
      assertAlive();
      throw error;
    }
  };
  const owned: OwnedApi = {
    baseUrl,
    process,
    fetch: guardedFetch,
    assertAlive,
    onExit(listener) {
      if (exitError) {
        listener(exitError);
        return () => {};
      }
      exitListeners.add(listener);
      return () => exitListeners.delete(listener);
    },
    async stop() {
      if (process.exitCode === null && !process.killed) process.kill("SIGTERM");
    },
  };

  const started = now();
  while (now() - started < timeoutMs) {
    assertAlive();
    const remaining = Math.max(1, timeoutMs - (now() - started));
    let response: Response | undefined;
    try {
      response = await guardedFetch(baseUrl, {
        signal: AbortSignal.timeout(Math.min(1000, remaining)),
      });
    } catch (error) {
      if (exitError) throw exitError;
    }
    assertAlive();
    if (response?.ok) {
      let identity: any;
      try {
        identity = await response.json();
      } catch {
        identity = undefined;
      }
      if (identity?.name !== "Argus Engine API" || identity?.status !== "ok") {
        await owned.stop();
        throw new Error(
          `Service on port ${port} is not the owned Argus Engine API. Check runs/desktop/frontend.log.`,
        );
      }
      assertAlive();
      return owned;
    }
    await sleep(250);
  }
  await owned.stop();
  throw new Error(
    `Argus API did not become ready within ${timeoutMs / 1000} seconds. Check runs/desktop/frontend.log.`,
  );
}
