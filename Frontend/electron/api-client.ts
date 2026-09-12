export type ApiError = Error & {
  status?: number;
  code?: string;
  permission?: string;
};
export type PushEvent = {
  type: "health" | "triage" | "alert.new" | "alert.update";
  data: unknown;
};

type Operation = {
  method: "GET" | "POST" | "PUT" | "DELETE";
  path: (args: unknown[]) => string;
  body?: (args: unknown[]) => unknown;
  query?: (args: unknown[]) => Record<string, unknown>;
  normalize?: (value: any, client: ArgusApiClient) => unknown;
};
const fixed = (path: string) => () => path;
const item =
  (prefix: string, suffix = "") =>
  (args: unknown[]) =>
    `${prefix}/${encodeURIComponent(String(args[0]))}${suffix}`;

function normalizeIncident(value: any) {
  const state =
    value?.triage?.state ?? value?.triage_state ?? value?.review ?? "new";
  return {
    ...value,
    evidence_dir: value?.evidence?.dir ?? value?.evidence_dir,
    review: value?.review ?? state,
    triage_state:
      state === "ack"
        ? "acknowledged"
        : ["true", "false"].includes(state)
          ? "resolved"
          : state,
  };
}

function loopbackUrl(value: unknown, protocols: string[]): string {
  if (typeof value !== "string") throw new Error("unsafe stream descriptor URL");
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error("unsafe stream descriptor URL");
  }
  if (
    !protocols.includes(parsed.protocol) ||
    !["127.0.0.1", "localhost", "::1", "[::1]"].includes(parsed.hostname) ||
    parsed.username ||
    parsed.password
  )
    throw new Error("unsafe stream descriptor URL");
  return parsed.toString();
}

function normalizeStreamDescriptor(value: any) {
  if (!value || typeof value !== "object")
    throw new Error("unsafe stream descriptor shape");
  if (value.kind === "mjpeg")
    return { kind: "mjpeg", url: loopbackUrl(value.url, ["http:"]) };
  if (value.kind !== "webrtc")
    throw new Error("unsafe stream descriptor kind");
  const normalized: {
    kind: "webrtc";
    url: string;
    mjpeg_fallback?: string;
  } = {
    kind: "webrtc",
    url: loopbackUrl(value.url, ["http:"]),
  };
  if (value.mjpeg_fallback !== undefined && value.mjpeg_fallback !== null)
    normalized.mjpeg_fallback = loopbackUrl(value.mjpeg_fallback, ["http:"]);
  if (value.ws !== undefined && value.ws !== null)
    loopbackUrl(value.ws, ["ws:"]);
  return normalized;
}

const operations: Record<string, Operation> = {
  get_site: { method: "GET", path: fixed("/site") },
  set_site: {
    method: "PUT",
    path: fixed("/site"),
    body: ([name, notify]) => ({ name, notify }),
  },
  list_cameras: { method: "GET", path: fixed("/cameras") },
  add_camera: {
    method: "POST",
    path: fixed("/cameras"),
    body: ([camera]) => ({ camera }),
  },
  remove_camera: { method: "DELETE", path: item("/cameras") },
  test: {
    method: "POST",
    path: fixed("/cameras/probe"),
    body: ([source]) => ({ source }),
  },
  discover_cameras: { method: "GET", path: fixed("/cameras/discovery") },
  detect_subnet: { method: "GET", path: fixed("/cameras/discovery/subnet") },
  scan: {
    method: "POST",
    path: fixed("/cameras/discovery/scan"),
    body: ([cidr]) => ({ cidr }),
  },
  list_areas: { method: "GET", path: fixed("/areas") },
  create_area: {
    method: "POST",
    path: fixed("/areas"),
    body: ([area]) => ({ area }),
  },
  assign_camera_area: {
    method: "PUT",
    path: item("/cameras", "/area"),
    body: ([, area_id]) => ({ area_id }),
  },
  scene_context: { method: "GET", path: item("/cameras", "/scene") },
  scene_review_summary: { method: "GET", path: fixed("/scene-mapping/review") },
  area_context: { method: "GET", path: item("/areas", "/context") },
  approve_area_context: {
    method: "POST",
    path: item("/areas", "/context/approve"),
    body: ([, context]) => ({ context }),
  },
  approve_site_context: {
    method: "POST",
    path: fixed("/site/context/approve"),
    body: ([context]) => ({ context }),
  },
  update_scene_context: {
    method: "PUT",
    path: item("/cameras", "/scene"),
    body: ([, context]) => ({ context }),
  },
  approve_scene_context: {
    method: "POST",
    path: item("/cameras", "/scene/approve"),
    body: ([, context]) => ({ context }),
  },
  request_scene_remap: {
    method: "POST",
    path: item("/cameras", "/scene/remap"),
  },
  enqueue_scene_mapping: {
    method: "POST",
    path: fixed("/scene-mapping/queue"),
    body: ([camera_ids]) => ({ camera_ids }),
  },
  scene_mapping_progress: {
    method: "GET",
    path: fixed("/scene-mapping/progress"),
  },
  accept_suggested_zone: {
    method: "POST",
    path: ([camera, name]) =>
      `/cameras/${encodeURIComponent(String(camera))}/zones/suggestions/${encodeURIComponent(String(name))}/accept`,
    body: ([, , dwell_seconds]) => ({ dwell_seconds }),
  },
  camera_snapshot: { method: "GET", path: item("/cameras", "/snapshot") },
  camera_stream: {
    method: "GET",
    path: item("/cameras", "/stream"),
    normalize: normalizeStreamDescriptor,
  },
  list_zones: { method: "GET", path: item("/cameras", "/zones") },
  add_zone: {
    method: "POST",
    path: item("/cameras", "/zones"),
    body: ([, name, points, dwell_seconds]) => ({
      name,
      points,
      dwell_seconds,
    }),
  },
  remove_zone: {
    method: "DELETE",
    path: ([camera, name]) =>
      `/cameras/${encodeURIComponent(String(camera))}/zones/${encodeURIComponent(String(name))}`,
  },
  set_camera_rules: {
    method: "PUT",
    path: item("/cameras", "/rules"),
    body: ([, rules]) => ({ rules }),
  },
  presets: { method: "GET", path: fixed("/cameras/presets") },
  use_case_templates: { method: "GET", path: fixed("/site/templates") },
  apply_template: {
    method: "POST",
    path: ([name]) =>
      `/site/templates/${encodeURIComponent(String(name))}/apply`,
  },
  add_custom_rule: {
    method: "POST",
    path: item("/cameras", "/rules/custom"),
    body: ([, question, dwell]) => ({ question, dwell }),
  },
  remove_custom_rule: {
    method: "DELETE",
    path: ([camera, name]) =>
      `/cameras/${encodeURIComponent(String(camera))}/rules/custom/${encodeURIComponent(String(name))}`,
  },
  english_rules_status: { method: "GET", path: fixed("/rules/english/status") },
  list_events: {
    method: "GET",
    path: fixed("/events"),
    query: ([limit]) => ({ limit: limit ?? 100 }),
    normalize: (value, client) =>
      (Array.isArray(value) ? value : (value?.events ?? [])).map((event: any) =>
        client.rememberIncident(event),
      ),
  },
  acknowledge_alert: { method: "POST", path: item("/events", "/acknowledge") },
  resolve_alert: {
    method: "POST",
    path: item("/events", "/resolve"),
    body: ([, outcome, note]) => ({ outcome, note }),
  },
  search_events: {
    method: "GET",
    path: fixed("/events"),
    query: ([q, limit]) => ({ q, limit: limit ?? 200 }),
  },
  camera_links: { method: "GET", path: item("/cameras", "/links") },
  start_monitoring: { method: "POST", path: fixed("/engine/start") },
  stop_monitoring: { method: "POST", path: fixed("/engine/stop") },
  monitoring_status: { method: "GET", path: fixed("/monitor") },
  feed_sources: { method: "GET", path: fixed("/engine/feeds") },
  switch_feed: {
    method: "POST",
    path: fixed("/engine/feeds/switch"),
    body: ([key]) => ({ key }),
  },
  feed_switch_status: { method: "GET", path: fixed("/engine/feeds/switch") },
  setup_state: { method: "GET", path: fixed("/setup/state") },
  setup_check: { method: "GET", path: fixed("/setup/check") },
  mark_configured: { method: "POST", path: fixed("/setup/configured") },
  send_test_notification: {
    method: "POST",
    path: fixed("/site/notifications/test"),
  },
  gate_status: {
    method: "GET",
    path: fixed("/engine/gate"),
    query: ([model]) => ({ model }),
  },
  pull_model: {
    method: "POST",
    path: fixed("/engine/models/pull"),
    body: ([model]) => ({ model }),
  },
  pull_progress: {
    method: "GET",
    path: fixed("/engine/models/pull"),
    query: ([model]) => ({ model }),
  },
  retention_status: { method: "GET", path: fixed("/retention") },
  set_retention: {
    method: "PUT",
    path: fixed("/retention"),
    body: ([days]) => ({ days }),
  },
  list_users: { method: "GET", path: fixed("/users") },
  add_user: {
    method: "POST",
    path: fixed("/users"),
    body: ([username, password, role]) => ({ username, password, role }),
  },
  remove_user: { method: "DELETE", path: item("/users") },
  audit_entries: {
    method: "GET",
    path: fixed("/audit"),
    query: ([limit]) => ({ limit }),
  },
  backup_now: { method: "POST", path: fixed("/backups") },
  download_diagnostics: { method: "POST", path: fixed("/diagnostics/bundle") },
  value_summary: {
    method: "GET",
    path: fixed("/value/summary"),
    query: ([days]) => ({ days }),
  },
  role_table: { method: "GET", path: fixed("/roles") },
  disk_encryption: { method: "GET", path: fixed("/system/disk-encryption") },
  organization: { method: "GET", path: fixed("/organization") },
  update_organization: {
    method: "PUT",
    path: fixed("/organization"),
    body: ([organization]) => ({ organization }),
  },
  list_branches: { method: "GET", path: fixed("/branches") },
  create_branch: {
    method: "POST",
    path: fixed("/branches"),
    body: ([branch]) => ({ branch }),
  },
  update_branch: {
    method: "PUT",
    path: item("/branches"),
    body: ([, branch]) => ({ branch }),
  },
  remove_branch: { method: "DELETE", path: item("/branches") },
  hierarchy: { method: "GET", path: fixed("/hierarchy") },
};

export const API_OPERATIONS = Object.freeze(Object.keys(operations));

export class ArgusApiClient {
  private readonly fetchImpl: typeof fetch;
  private readonly WebSocketImpl?: typeof WebSocket;
  private readonly listeners = new Set<(event: PushEvent) => void>();
  private readonly evidenceIds = new Map<string, string>();
  private token?: string;
  private socket?: WebSocket;
  private reconnectTimer?: ReturnType<typeof setTimeout>;
  private reconnectAttempt = 0;
  private closed = false;

  constructor(
    private readonly baseUrl: string,
    options: {
      fetch?: typeof fetch;
      WebSocket?: typeof WebSocket;
      onToken?: (token: string) => void;
    } = {},
  ) {
    this.fetchImpl = options.fetch ?? fetch;
    this.WebSocketImpl = options.WebSocket ?? globalThis.WebSocket;
    this.onToken = options.onToken;
  }

  private readonly onToken?: (token: string) => void;

  async invoke<T>(method: string, args: unknown[] = []): Promise<T> {
    if (!Array.isArray(args)) throw new Error("Invalid operation arguments");
    if (method === "auth_state") return (await this.authState()) as T;
    if (method === "sign_in") return (await this.signIn(args)) as T;
    if (method === "create_first_owner") {
      await this.request(
        {
          method: "POST",
          path: fixed("/auth/first-owner"),
          body: ([username, password]) => ({ username, password }),
        },
        args,
        false,
      );
      return (await this.signIn(args)) as T;
    }
    if (method === "sign_out") {
      const result = this.token
        ? await this.request(
            { method: "DELETE", path: fixed("/auth/session") },
            [],
            true,
          )
        : { ok: true };
      this.clearSession();
      return (result ?? { ok: true }) as T;
    }
    if (method === "event_clip") {
      const key = String(args[0] ?? "");
      return (await this.request(
        { method: "GET", path: item("/events", "/clip") },
        [this.evidenceIds.get(key) ?? key],
        true,
      )) as T;
    }
    if (method === "live_start" || method === "live_stop")
      throw new Error(
        "Live streams are resolved per camera through camera_stream.",
      );
    const operation = operations[method];
    if (!operation) throw new Error(`Unsupported engine operation: ${method}`);
    return (await this.request(operation, args, true)) as T;
  }

  subscribe(listener: (event: PushEvent) => void): () => void {
    this.listeners.add(listener);
    this.connectSocket();
    return () => {
      this.listeners.delete(listener);
      if (!this.listeners.size) this.stopSocket();
    };
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.token) {
      try {
        await this.request(
          { method: "DELETE", path: fixed("/auth/session") },
          [],
          true,
        );
      } catch {
        /* API may already be gone. */
      }
    }
    this.clearSession();
    this.listeners.clear();
  }

  rememberIncident(value: any) {
    const event = normalizeIncident(value);
    if (event.evidence_dir)
      this.evidenceIds.set(String(event.evidence_dir), String(event.id));
    return event;
  }

  private async authState() {
    const state = (await this.request(
      { method: "GET", path: fixed("/auth/state") },
      [],
      false,
    )) as any;
    if (!this.token)
      return {
        ...state,
        signed_in: false,
        username: "",
        role: "",
        permissions: [],
      };
    try {
      const user = (await this.request(
        { method: "GET", path: fixed("/auth/me") },
        [],
        true,
      )) as any;
      return { ...state, signed_in: true, ...user };
    } catch (error) {
      if ((error as ApiError).status !== 401) throw error;
      this.clearSession();
      return {
        ...state,
        signed_in: false,
        username: "",
        role: "",
        permissions: [],
      };
    }
  }

  private async signIn(args: unknown[]) {
    const value = (await this.request(
      {
        method: "POST",
        path: fixed("/auth/session"),
        body: ([username, password]) => ({ username, password }),
      },
      args,
      false,
    )) as any;
    this.token = value.token;
    this.onToken?.(value.token);
    this.closed = false;
    this.connectSocket();
    return {
      configured: true,
      signed_in: true,
      username: value.user.username,
      role: value.user.role,
      permissions: value.user.permissions,
    };
  }

  private async request(
    operation: Operation,
    args: unknown[],
    authenticated: boolean,
  ) {
    const url = new URL(this.baseUrl + operation.path(args));
    for (const [key, value] of Object.entries(operation.query?.(args) ?? {}))
      if (value !== undefined && value !== null && value !== "")
        url.searchParams.set(key, String(value));
    const headers = new Headers();
    if (authenticated) {
      if (!this.token)
        throw Object.assign(new Error("Sign in to access the local engine"), {
          status: 401,
          code: "unauthorized",
        });
      headers.set("authorization", `Bearer ${this.token}`);
    }
    const body = operation.body?.(args);
    if (body !== undefined) headers.set("content-type", "application/json");
    const response = await this.fetchImpl(url, {
      method: operation.method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await response.text();
    let value: any = null;
    if (text) {
      try {
        value = JSON.parse(text);
      } catch {
        value = text;
      }
    }
    if (!response.ok) {
      const envelope = value?.error ?? {};
      const permission = envelope?.detail?.permission;
      const message =
        envelope.message ??
        value?.detail ??
        response.statusText ??
        "API request failed";
      throw Object.assign(
        new Error(
          permission ? `${message} (requires ${permission})` : String(message),
        ),
        { status: response.status, code: envelope.code, permission },
      ) as ApiError;
    }
    return operation.normalize ? operation.normalize(value, this) : value;
  }

  private connectSocket() {
    if (
      this.closed ||
      !this.token ||
      !this.listeners.size ||
      this.socket ||
      !this.WebSocketImpl
    )
      return;
    const url = new URL(this.baseUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = `${url.pathname.replace(/\/$/, "")}/stream`;
    url.search = "";
    const socket = new this.WebSocketImpl(url.toString(), [
      "argus.v1",
      `argus.token.${this.token}`,
    ]);
    this.socket = socket;
    socket.onopen = () => {
      this.reconnectAttempt = 0;
    };
    socket.onmessage = (message) => {
      try {
        const parsed = JSON.parse(String(message.data));
        if (!isPushEvent(parsed)) return;
        const event = parsed.type.startsWith("alert.")
          ? ({
              type: parsed.type,
              data: this.rememberIncident(parsed.data),
            } as PushEvent)
          : ({ type: parsed.type, data: parsed.data } as PushEvent);
        for (const listener of this.listeners) listener(event);
      } catch {
        /* Recovery polling covers malformed push data. */
      }
    };
    socket.onerror = () => socket.close();
    socket.onclose = (event) => {
      if (this.socket === socket) this.socket = undefined;
      if ([4401, 4403].includes(event.code)) {
        if (event.code === 4401) {
          this.token = undefined;
          this.evidenceIds.clear();
        }
        if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
        this.reconnectTimer = undefined;
        return;
      }
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect() {
    if (
      this.closed ||
      !this.token ||
      !this.listeners.size ||
      this.reconnectTimer
    )
      return;
    const delay = Math.min(1000 * 2 ** this.reconnectAttempt++, 30000);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined;
      this.connectSocket();
    }, delay);
  }

  private stopSocket() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = undefined;
    const socket = this.socket;
    this.socket = undefined;
    socket?.close();
  }

  private clearSession() {
    this.token = undefined;
    this.evidenceIds.clear();
    this.stopSocket();
  }
}

function isPushEvent(value: any): value is PushEvent {
  return Boolean(
    value &&
    typeof value === "object" &&
    ["health", "triage", "alert.new", "alert.update"].includes(value.type) &&
    "data" in value,
  );
}
