import { afterEach, describe, expect, it, vi } from "vitest";
import { ArgusApiClient } from "../electron/api-client.js";

type FetchCall = { url: string; init: RequestInit };

function response(body: unknown, status = 200): Response {
  return new Response(status === 204 ? null : JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function fetchSequence(items: Response[]) {
  const calls: FetchCall[] = [];
  const fetch = vi.fn(async (input: string | URL | Request, init = {}) => {
    calls.push({ url: String(input), init });
    const next = items.shift();
    if (!next) throw new Error("unexpected fetch");
    return next;
  }) as unknown as typeof globalThis.fetch;
  return { fetch, calls };
}

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  readonly url: string;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  close() {
    this.closed = true;
  }

  emit(type: string, data: unknown) {
    this.onmessage?.({ data: JSON.stringify({ type, ts: 1, data }) });
  }

  disconnect() {
    this.onclose?.();
  }
}

describe("ArgusApiClient", () => {
  afterEach(() => {
    FakeWebSocket.instances = [];
    vi.useRealTimers();
  });

  it("keeps the token in the client and sends the bearer header", async () => {
    const net = fetchSequence([
      response({
        token: "main-process-secret",
        expires_at: "2026-09-12T00:00:00Z",
        user: { username: "ayo", role: "owner", permissions: ["view_live"] },
      }),
      response([]),
    ]);
    const client = new ArgusApiClient("http://127.0.0.1:8787/api/v1", {
      fetch: net.fetch,
    });

    const auth = await client.invoke("sign_in", ["ayo", "correct horse"]);
    await client.invoke("list_cameras");

    expect(auth).toEqual({
      configured: true,
      signed_in: true,
      username: "ayo",
      role: "owner",
      permissions: ["view_live"],
    });
    expect(net.calls[0].init.body).toBe(
      JSON.stringify({ username: "ayo", password: "correct horse" }),
    );
    expect(new Headers(net.calls[1].init.headers).get("authorization")).toBe(
      "Bearer main-process-secret",
    );
    expect(JSON.stringify(auth)).not.toContain("main-process-secret");
  });

  it("reports the actual token only to an internal observer", async () => {
    const net = fetchSequence([
      response({
        token: "random-main-token",
        user: { username: "ayo", role: "owner", permissions: [] },
      }),
    ]);
    const onToken = vi.fn();
    const client = new ArgusApiClient("http://127.0.0.1:8787/api/v1", {
      fetch: net.fetch,
      onToken,
    });

    const auth = await client.invoke("sign_in", ["ayo", "secret"]);

    expect(onToken).toHaveBeenCalledOnce();
    expect(onToken).toHaveBeenCalledWith("random-main-token");
    expect(JSON.stringify(auth)).not.toContain("random-main-token");
  });

  it("maps list_events envelope to the existing incident array", async () => {
    const net = fetchSequence([
      response({
        token: "t",
        user: { username: "a", role: "owner", permissions: [] },
      }),
      response({
        events: [
          {
            id: "evt_7",
            ts: 7,
            camera_id: "front",
            rule: "loitering",
            priority: "high",
            confidence: 0.9,
            reason: "waiting",
            verdict: "confirmed",
            evidence: { dir: "/private/e7", thumb: "/thumb", clip: true },
            triage: { state: "ack" },
          },
        ],
        next_cursor: null,
      }),
      response({ uri: "data:video/mp4;base64,AA==", frames: [] }),
    ]);
    const client = new ArgusApiClient("http://127.0.0.1:8787/api/v1", {
      fetch: net.fetch,
    });
    await client.invoke("sign_in", ["a", "b"]);

    const events = await client.invoke<any[]>("list_events", [100]);
    await client.invoke("event_clip", ["/private/e7"]);

    expect(events[0]).toMatchObject({
      id: "evt_7",
      evidence_dir: "/private/e7",
      review: "ack",
      triage_state: "acknowledged",
    });
    expect(net.calls[1].url).toBe(
      "http://127.0.0.1:8787/api/v1/events?limit=100",
    );
    expect(net.calls[2].url).toBe(
      "http://127.0.0.1:8787/api/v1/events/evt_7/clip",
    );
  });

  it("combines public auth state with auth me", async () => {
    const net = fetchSequence([
      response({
        token: "t",
        user: { username: "ayo", role: "owner", permissions: [] },
      }),
      response({ configured: true }),
      response({
        username: "ayo",
        role: "owner",
        permissions: ["view_alerts"],
      }),
    ]);
    const client = new ArgusApiClient("http://127.0.0.1:8787/api/v1", {
      fetch: net.fetch,
    });
    await client.invoke("sign_in", ["ayo", "pw"]);

    await expect(client.invoke("auth_state")).resolves.toEqual({
      configured: true,
      signed_in: true,
      username: "ayo",
      role: "owner",
      permissions: ["view_alerts"],
    });
  });

  it("surfaces the missing permission from a 403 envelope", async () => {
    const net = fetchSequence([
      response({
        token: "t",
        user: { username: "op", role: "operator", permissions: [] },
      }),
      response(
        {
          error: {
            code: "forbidden",
            message: "permission denied",
            detail: { permission: "configure_cameras" },
          },
        },
        403,
      ),
    ]);
    const client = new ArgusApiClient("http://127.0.0.1:8787/api/v1", {
      fetch: net.fetch,
    });
    await client.invoke("sign_in", ["op", "pw"]);

    await expect(
      client.invoke("add_camera", [{ id: "x", source: "demo" }]),
    ).rejects.toMatchObject({
      message: "permission denied (requires configure_cameras)",
      status: 403,
      code: "forbidden",
      permission: "configure_cameras",
    });
  });

  it("maps legacy invoke arguments to documented request bodies", async () => {
    const net = fetchSequence([
      response({
        token: "t",
        user: { username: "a", role: "owner", permissions: [] },
      }),
      response({ ok: true }),
      response({ ok: true }),
      response({ ok: true }),
    ]);
    const client = new ArgusApiClient("http://127.0.0.1:8787/api/v1", {
      fetch: net.fetch,
    });
    await client.invoke("sign_in", ["a", "pw"]);

    await client.invoke("set_site", ["Depot", "console"]);
    await client.invoke("add_zone", ["Front Door", "Till", [[0, 0]], 7]);
    await client.invoke("resolve_alert", ["evt_9", "false_alarm", "shadow"]);

    expect(
      net.calls
        .slice(1)
        .map((call) => [call.url, call.init.method, call.init.body]),
    ).toEqual([
      [
        "http://127.0.0.1:8787/api/v1/site",
        "PUT",
        JSON.stringify({ name: "Depot", notify: "console" }),
      ],
      [
        "http://127.0.0.1:8787/api/v1/cameras/Front%20Door/zones",
        "POST",
        JSON.stringify({ name: "Till", points: [[0, 0]], dwell_seconds: 7 }),
      ],
      [
        "http://127.0.0.1:8787/api/v1/events/evt_9/resolve",
        "POST",
        JSON.stringify({ outcome: "false_alarm", note: "shadow" }),
      ],
    ]);
  });

  it("hydrates subscribers and reconnects with capped exponential delays", async () => {
    vi.useFakeTimers();
    const net = fetchSequence([
      response({
        token: "secret",
        user: { username: "a", role: "owner", permissions: [] },
      }),
    ]);
    const client = new ArgusApiClient("http://127.0.0.1:8787/api/v1", {
      fetch: net.fetch,
      WebSocket: FakeWebSocket as unknown as typeof WebSocket,
    });
    const received: string[] = [];
    client.subscribe((event) => received.push(event.type));
    await client.invoke("sign_in", ["a", "pw"]);

    const first = FakeWebSocket.instances[0];
    expect(first.url).toBe("ws://127.0.0.1:8787/api/v1/stream?token=secret");
    first.emit("health", { status: "ok" });
    first.emit("triage", { to_review: 1 });
    first.emit("alert.new", { id: "evt_1" });
    first.emit("alert.update", { id: "evt_1", review: "ack" });
    expect(received).toEqual(["health", "triage", "alert.new", "alert.update"]);

    first.disconnect();
    for (const delay of [1000, 2000, 4000, 8000, 16000, 30000]) {
      await vi.advanceTimersByTimeAsync(delay - 1);
      const count = FakeWebSocket.instances.length;
      await vi.advanceTimersByTimeAsync(1);
      expect(FakeWebSocket.instances).toHaveLength(count + 1);
      FakeWebSocket.instances.at(-1)?.disconnect();
    }

    await client.close();
    const before = FakeWebSocket.instances.length;
    await vi.advanceTimersByTimeAsync(60000);
    expect(FakeWebSocket.instances).toHaveLength(before);
  });
});
