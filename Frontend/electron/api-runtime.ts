import { ArgusApiClient } from "./api-client.js";
import type { OwnedApi } from "./api-supervisor.js";

export function createOwnedApiClient(owner: OwnedApi) {
  owner.assertAlive();
  const client = new ArgusApiClient(owner.baseUrl, { fetch: owner.fetch });
  owner.assertAlive();
  return client;
}
