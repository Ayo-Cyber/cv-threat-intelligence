import { describe, expect, it, vi } from "vitest";
import { ArgusApiClient } from "../electron/api-client.js";

describe("camera probe request mapping", () => {
  it("sends the frozen canonical source field", async () => {
    const calls: RequestInit[] = [];
    const fetch = vi.fn(async (_input: string | URL | Request, init = {}) => {
      calls.push(init);
      if (calls.length === 1)
        return new Response(
          JSON.stringify({
            token: "secret",
            user: { username: "ayo", role: "owner", permissions: [] },
          }),
        );
      return new Response(JSON.stringify({ ok: true }));
    }) as unknown as typeof globalThis.fetch;
    const client = new ArgusApiClient("http://127.0.0.1:8787/api/v1", {
      fetch,
    });
    await client.invoke("sign_in", ["ayo", "pw"]);

    await client.invoke("test", ["rtsp://camera/live"]);

    expect(calls[1].body).toBe(
      JSON.stringify({ source: "rtsp://camera/live" }),
    );
  });
});
