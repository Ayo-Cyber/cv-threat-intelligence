import type {
  Area,
  Branch,
  Camera,
  Hierarchy,
  Incident,
  Json,
  Organization,
  Scene,
  Transport,
  Zone,
} from "./types";
const KEY = "argus.desktop.demo.v1";
const MUTATING_METHODS = new Set([
  "set_site",
  "create_area",
  "assign_camera_area",
  "start_monitoring",
  "stop_monitoring",
  "update_scene_context",
  "approve_scene_context",
  "add_zone",
  "remove_zone",
  "set_camera_rules",
  "add_custom_rule",
  "remove_custom_rule",
  "acknowledge_alert",
  "resolve_alert",
  "add_camera",
  "remove_camera",
  "mark_configured",
  "apply_template",
  "reset_demo",
]);
interface DemoState {
  organization: Organization;
  branches: Omit<Branch, "areas">[];
  cameras: Camera[];
  events: Incident[];
  scenes: Record<string, Scene>;
  zones: Record<string, Zone[]>;
  areas: Omit<Area, "cameras">[];
  site: Json;
  running: boolean;
  configured: boolean;
}
export function initialDemo(): DemoState {
  const cameras: Camera[] = [
    {
      id: "Loading Bay",
      source: "demo/empty_warehouse.mp4",
      demo_video: "demo/empty_warehouse.mp4",
      snapshot: "demo/empty_warehouse-frame.jpg",
      area_id: "warehouse",
      branch_id: "lagos-plant",
      area: "Warehouse",
      crowd_formation: true,
    },
    {
      id: "Forecourt ATM",
      source: "demo/theft_yt_01.mp4",
      demo_video: "demo/theft_yt_01.mp4",
      snapshot: "demo/theft_yt_01-frame.jpg",
      area_id: "external",
      branch_id: "ikeja-outlet",
      area: "External perimeter",
      weapons: true,
    },
    {
      id: "Retail Aisle",
      source: "demo/theft_shop_01.mp4",
      demo_video: "demo/theft_shop_01.mp4",
      snapshot: "demo/theft_shop_01-frame.jpg",
      area_id: "retail",
      branch_id: "ikeja-outlet",
      area: "Retail floor",
      concealment: true,
    },
    {
      id: "Main Corridor",
      source: "demo/normal_street_01.mp4",
      demo_video: "demo/normal_street_01.mp4",
      snapshot: "demo/normal_street_01-frame.jpg",
      area_id: "external",
      branch_id: "ikeja-outlet",
      area: "Public access",
      running: false,
    },
  ];
  const scenes: Record<string, Scene> = {};
  cameras.forEach(
    (c, i) =>
      (scenes[c.id] = {
        environment_type: ["warehouse", "forecourt", "retail", "public_area"][
          i
        ],
        scene_description: [
          "Indoor warehouse with storage space, a raised platform and access routes.",
          "Outdoor forecourt with a payment kiosk and vehicle access.",
          "Retail shop with shelves, merchandise and customer aisles.",
          "Open pedestrian circulation area.",
        ][i],
        expected_actors: ["person"],
        confidence: 0.86,
        source_frame_uri: c.snapshot,
        mapping: {
          status: i === 0 ? "ready_unreviewed" : "ready_reviewed",
          provenance: "demo",
          reviewed_by: i === 0 ? "" : "Sample operator",
        },
        zones: [],
      }),
  );
  return {
    organization: { id: "deluxe-paints", name: "Deluxe Paints Nigeria" },
    branches: [
      { id: "lagos-plant", name: "Lagos Plant" },
      { id: "ikeja-outlet", name: "Ikeja Outlet" },
    ],
    cameras,
    scenes,
    zones: {},
    areas: [
      {
        id: "warehouse",
        name: "Warehouse",
        branch_id: "lagos-plant",
      },
      {
        id: "external",
        name: "External perimeter",
        branch_id: "ikeja-outlet",
      },
      { id: "retail", name: "Retail floor", branch_id: "ikeja-outlet" },
    ],
    site: {
      name: "Deluxe Paints Nigeria",
      notify: "console",
      configured: true,
    },
    running: false,
    configured: true,
    events: [
      {
        id: "sample-1",
        camera_id: "Forecourt ATM",
        title: "ATM interference",
        rule: "custom:atm_break_in",
        priority: "high",
        ts: 1788783128,
        reason:
          "Sample incident: inspect the activity around the payment kiosk and record your assessment.",
        review: "new",
        verdict: "unverified",
        demo_video: "demo/theft_yt_01.mp4",
      },
      {
        id: "sample-2",
        camera_id: "Retail Aisle",
        title: "Possible concealment",
        rule: "concealment",
        priority: "medium",
        ts: 1788782921,
        reason:
          "Sample incident: review the shopper and merchandise interaction. This is a UI fixture, not a model verdict.",
        review: "new",
        verdict: "unverified",
        demo_video: "demo/theft_shop_01.mp4",
      },
    ],
  };
}

function demoHierarchy(state: DemoState): Hierarchy {
  const areaIds = new Set(state.areas.map((area) => area.id));
  return {
    organization: state.organization,
    branches: state.branches.map((branch) => ({
      ...branch,
      areas: state.areas
        .filter((area) => area.branch_id === branch.id)
        .map((area) => ({
          ...area,
          cameras: state.cameras.filter((camera) => camera.area_id === area.id),
        })),
    })),
    unassigned_cameras: state.cameras.filter(
      (camera) => !camera.area_id || !areaIds.has(camera.area_id),
    ),
  };
}

export function createDemo(
  storage?: Pick<Storage, "getItem" | "setItem">,
): Transport {
  let state = initialDemo();
  try {
    const raw = storage?.getItem(KEY);
    if (raw) {
      const saved = JSON.parse(raw);
      if (
        Array.isArray(saved.cameras) &&
        saved.scenes &&
        Array.isArray(saved.events)
      ) {
        // Upgrade coordinates persisted by the first browser prototype.
        for (const zones of Object.values(saved.zones || {}) as Json[][]) {
          for (const zone of zones) {
            if (!zone.polygon && Array.isArray(zone.points)) {
              zone.polygon = zone.points;
              delete zone.points;
            }
          }
        }
        const branches = saved.branches || state.branches;
        const defaultBranchByArea = new Map(
          state.areas.map((area) => [area.id, area.branch_id]),
        );
        const areas = (saved.areas || state.areas).map((area: Json) => ({
          ...area,
          branch_id:
            area.branch_id ||
            defaultBranchByArea.get(area.id) ||
            branches[0]?.id,
        }));
        const branchByArea = new Map(
          areas.map((area: Json) => [area.id, area.branch_id]),
        );
        state = {
          ...state,
          ...saved,
          organization: saved.organization || state.organization,
          branches,
          areas,
          cameras: saved.cameras.map((camera: Camera) => ({
            ...camera,
            branch_id: branchByArea.get(camera.area_id || ""),
          })),
        };
      }
    }
  } catch {
    /* Invalid demo data starts a fresh review workspace. */
  }
  const save = () => storage?.setItem(KEY, JSON.stringify(state));
  return {
    async invoke<T>(method: string, args: unknown[] = []): Promise<T> {
      const [id, a, b, c] = args as any[];
      let result: any = { ok: true };
      const cam = state.cameras.find((v) => v.id === id);
      switch (method) {
        case "auth_state":
          result = {
            configured: true,
            signed_in: true,
            username: "Demi O.",
            role: "owner",
            permissions: [
              "configure_cameras",
              "configure_detectors",
              "control_engine",
              "view_alerts",
              "review_alerts",
            ],
          };
          break;
        case "get_site":
          result = state.site;
          break;
        case "organization":
          return structuredClone(state.organization) as T;
        case "list_branches":
          return structuredClone(state.branches) as T;
        case "hierarchy":
          return structuredClone(demoHierarchy(state)) as T;
        case "set_site":
          state.site = {
            ...state.site,
            name: id,
            notify: a || state.site.notify,
          };
          break;
        case "list_cameras":
          result = state.cameras;
          break;
        case "list_events":
          result = state.events;
          break;
        case "list_areas":
          result = state.areas;
          break;
        case "create_area": {
          const area = { ...id, id: id.id || `area-${Date.now()}` };
          state.areas.push(area);
          result = area;
          break;
        }
        case "assign_camera_area":
          if (cam) cam.area_id = a;
          break;
        case "monitoring_status":
          result = {
            running: state.running,
            phase: state.running ? "demo_playback" : "stopped",
            demo: true,
          };
          break;
        case "start_monitoring":
        case "stop_monitoring":
          state.running = method === "start_monitoring";
          result = { running: state.running, demo: true };
          break;
        case "scene_context":
          result = state.scenes[id] || null;
          break;
        case "update_scene_context":
        case "approve_scene_context":
          state.scenes[id] = {
            ...state.scenes[id],
            ...a,
            mapping: {
              status:
                method === "approve_scene_context"
                  ? "ready_reviewed"
                  : "ready_unreviewed",
              provenance: "demo",
              reviewed_by: method === "approve_scene_context" ? "Demi O." : "",
            },
          };
          result = { context: state.scenes[id] };
          break;
        case "request_scene_remap":
        case "enqueue_scene_mapping":
          throw new Error(
            "AI remapping requires the local engine. Demo mode does not generate scene evidence.",
          );
        case "list_zones":
          result = state.zones[id] || [];
          break;
        case "add_zone": {
          const z = { name: a, polygon: b, dwell_alert_seconds: c };
          state.zones[id] = [
            ...(state.zones[id] || []).filter((z) => z.name !== a),
            z,
          ];
          break;
        }
        case "remove_zone":
          state.zones[id] = (state.zones[id] || []).filter((z) => z.name !== a);
          break;
        case "set_camera_rules":
          if (cam) Object.assign(cam, a);
          break;
        case "add_custom_rule":
          if (cam) {
            cam.custom_rules = [
              ...(cam.custom_rules || []).filter((r) => r.question !== a),
              { question: a, dwell: b },
            ];
          }
          break;
        case "remove_custom_rule":
          if (cam)
            cam.custom_rules = (cam.custom_rules || []).filter(
              (r) => r.question !== a,
            );
          break;
        case "acknowledge_alert": {
          const e = state.events.find((e) => e.id === id);
          if (e) {
            e.review = "ack";
            e.triage_state = "acknowledged";
          }
          break;
        }
        case "resolve_alert": {
          const e = state.events.find((e) => e.id === id);
          if (e) {
            e.review =
              a === "false_alarm"
                ? "false"
                : a === "inconclusive"
                  ? "ack"
                  : "true";
            e.note = b;
            e.triage_state = "resolved";
          }
          break;
        }
        case "camera_snapshot":
          result = { uri: cam?.snapshot, w: 640, h: 480 };
          break;
        case "add_camera": {
          if (!id.id || state.cameras.some((c) => c.id === id.id))
            throw new Error("Use a unique camera name.");
          state.cameras.push({ ...id });
          result = state.cameras;
          break;
        }
        case "remove_camera":
          state.cameras = state.cameras.filter((c) => c.id !== id);
          delete state.scenes[id];
          break;
        case "english_rules_status":
          result = { available: false, demo: true };
          break;
        case "search_events":
          result = {
            results: state.events.filter((e) =>
              `${e.title} ${e.camera_id} ${e.reason}`
                .toLowerCase()
                .includes(String(id).toLowerCase()),
            ),
          };
          break;
        case "setup_state":
          result = { configured: state.configured };
          break;
        case "mark_configured":
          state.configured = true;
          break;
        case "use_case_templates":
          result = {
            retail: { label: "Retail" },
            manufacturing: { label: "Manufacturing" },
            warehouse: { label: "Warehouse" },
          };
          break;
        case "apply_template":
          state.cameras.forEach((c) => {
            c.fire_smoke = id === "manufacturing";
            c.concealment = id === "retail";
            c.crowd_formation = id === "warehouse";
          });
          break;
        case "presets":
          result = {};
          break;
        case "gate_status":
          result = { ready: false, demo: true };
          break;
        case "retention_status":
          result = { demo: true, days: 30 };
          break;
        case "set_retention":
          throw new Error("Evidence retention requires the local engine.");
        case "audit_entries":
          result = [];
          break;
        case "list_users":
          result = [{ username: "Demi O.", role: "owner" }];
          break;
        case "live_stop":
          break;
        case "reset_demo":
          state = initialDemo();
          break;
        case "feed_sources":
          result = { sources: [] };
          break;
        case "value_summary":
          result = { demo: true };
          break;
        default:
          throw new Error("This operation requires a connected Argus engine.");
      }
      if (MUTATING_METHODS.has(method)) save();
      return structuredClone(result) as T;
    },
  };
}
