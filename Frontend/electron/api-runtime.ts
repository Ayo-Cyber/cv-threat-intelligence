import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { ArgusApiClient } from "./api-client.js";
import type { OwnedApi } from "./api-supervisor.js";

type SmokeStateIo = Pick<typeof fs, "mkdirSync" | "writeFileSync">;

export function createSmokeStateWriter(
  env: Record<string, string | undefined>,
  io: SmokeStateIo = fs,
) {
  if (env.ARGUS_SMOKE_TEST !== "1") return undefined;
  const userData = path.resolve(
    env.ARGUS_USER_DATA || "ARGUS_USER_DATA-is-required",
  );
  const statePath = path.resolve(
    env.ARGUS_SMOKE_STATE || "ARGUS_SMOKE_STATE-is-required",
  );
  const relative = path.relative(userData, statePath);
  if (
    !env.ARGUS_USER_DATA ||
    !env.ARGUS_SMOKE_STATE ||
    !relative ||
    relative.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relative)
  )
    throw new Error("ARGUS_SMOKE_STATE must be inside ARGUS_USER_DATA.");
  const state: { api_pid?: number; token_sha256?: string } = {};
  const persist = () => {
    io.mkdirSync(path.dirname(statePath), { recursive: true });
    io.writeFileSync(statePath, JSON.stringify(state), {
      encoding: "utf8",
      mode: 0o600,
    });
  };
  return {
    recordPid(pid: number) {
      state.api_pid = pid;
      persist();
    },
    recordToken(token: string) {
      state.token_sha256 = createHash("sha256").update(token).digest("hex");
      persist();
    },
  };
}

export function createOwnedApiClient(
  owner: OwnedApi,
  options: { onToken?: (token: string) => void } = {},
) {
  owner.assertAlive();
  const client = new ArgusApiClient(owner.baseUrl, {
    fetch: owner.fetch,
    onToken: options.onToken,
  });
  owner.assertAlive();
  return client;
}
