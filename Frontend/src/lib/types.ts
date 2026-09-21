export type Json = Record<string, any>;
export type Mode = "demo" | "engine";
export type View =
  "watch" | "incidents" | "cameras" | "rules" | "settings" | "setup";
export type Point = [number, number];
export interface Zone {
  name: string;
  polygon: Point[];
  dwell_alert_seconds: number;
  kind?: string;
}
export interface Scene {
  environment_type: string;
  scene_description: string;
  expected_actors: string[];
  confidence: number;
  source_frame_uri?: string;
  zones?: Json[];
  zone_count?: number;
  mapping: { status: string; provenance?: string; reviewed_by?: string };
}
export interface Camera {
  id: string;
  source: string;
  area_id?: string;
  branch_id?: string;
  area?: string;
  config?: string;
  custom_rules?: { question: string; dwell: number }[];
  custom_threats?: { name: string; description: string }[];
  demo_video?: string;
  snapshot?: string;
  [key: string]: any;
}
export interface Organization {
  id: string;
  name: string;
}
export interface Area {
  id: string;
  name: string;
  branch_id: string;
  cameras: Camera[];
  implicit?: boolean;
  camera_ids?: string[];
}
export interface Branch {
  id: string;
  name: string;
  areas: Area[];
}
export interface Hierarchy {
  organization: Organization;
  branches: Branch[];
  unassigned_cameras: Camera[];
}
export interface Incident {
  id: string;
  camera_id: string;
  rule: string;
  priority: string;
  ts: number;
  reason: string;
  review: string;
  verdict?: string;
  evidence_dir?: string;
  demo_video?: string;
  confidence?: number;
  note?: string;
  title?: string;
  triage_state?: string;
}
export interface ObjectExample {
  id: string;
  source: string;
  bbox: [number, number, number, number];
  sha256: string;
  reviewed: boolean;
  bbox_format?: string;
}
export interface ObjectTarget {
  id: string;
  label: string;
  category: string;
  aliases: string[];
  review_state: "draft" | "active" | "degraded" | string;
  min_similarity: number;
  allowed_zone_ids: string[];
  examples: ObjectExample[];
  negative_examples: ObjectExample[];
  grounding_description?: string;
  can_activate?: boolean;
  ready_for_activation?: boolean;
  reasons?: string[];
  needs_reembed?: boolean;
  degraded_unavailable?: boolean;
}
export interface ObjectWatchRuntime {
  status: string;
  backend: string;
  fingerprint?: string | null;
  reason_codes: string[];
  structurally_available?: boolean;
  executable_verified?: boolean;
  dimensions?: number | null;
}
export interface ObjectWatchStatus {
  runtime?: ObjectWatchRuntime;
  targets: ObjectTarget[];
}
export interface ObjectWatchJobStatus {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | string;
  error?: string;
  message?: string;
}
export interface Auth {
  configured: boolean;
  signed_in: boolean;
  username: string;
  role: string;
  permissions: string[];
}
export interface Workspace {
  cameras: Camera[];
  events: Incident[];
  areas: Json[];
  hierarchy: Hierarchy;
  site: Json;
  monitor: Json;
  english: Json;
  auth: Auth;
}
export type StreamDescriptor =
  | {
      kind: "webrtc";
      url: string;
      ws?: string;
      mjpeg_fallback?: string | null;
    }
  | { kind: "mjpeg"; url: string };
export type CameraStreamArgs = [cameraId: string, tracking?: boolean];
export type PushEvent = {
  type: "health" | "triage" | "alert.new" | "alert.update";
  data: unknown;
};
export interface Transport {
  invoke<T = any>(method: string, args?: unknown[]): Promise<T>;
  subscribe?(listener: (event: PushEvent) => void): () => void;
}

declare global {
  interface Window {
    argusDesktop?: {
      invoke<T = any>(method: string, args?: unknown[]): Promise<T>;
      subscribe(listener: (event: PushEvent) => void): () => void;
      environment(): Promise<Json>;
    };
  }
}
export const DETECTORS = [
  {
    key: "concealment",
    name: "Concealment",
    group: "Security",
    detail: "Merchandise concealment candidates",
  },
  {
    key: "video_action",
    name: "Video theft",
    group: "Security",
    detail: "Requires the fine-tuned video checkpoint",
  },
  {
    key: "violence",
    name: "Violence",
    group: "Security",
    detail: "Experimental",
  },
  {
    key: "weapons",
    name: "Weapons",
    group: "Security",
    detail: "Requires compatible weapon weights",
  },
  {
    key: "theft",
    name: "Zone theft",
    group: "Security",
    detail: "Zone-based object interactions",
  },
  {
    key: "object_watch",
    name: "Object watchlists",
    group: "Security",
    detail: "Recognise enrolled Chi products and watched objects",
  },
  {
    key: "tamper",
    name: "Camera tampering",
    group: "Security",
    detail: "View obstruction and disruption",
  },
  {
    key: "fire_smoke",
    name: "Fire & smoke",
    group: "Safety / HSE",
    detail: "Visual fire and smoke candidates",
  },
  {
    key: "running",
    name: "Panic running",
    group: "Safety / HSE",
    detail: "Experimental; site calibration required",
  },
  {
    key: "crowd_formation",
    name: "Crowd formation",
    group: "Safety / HSE",
    detail: "Group density and proximity",
  },
  {
    key: "normal_movement",
    name: "Normal movement",
    group: "Operations",
    detail: "Moving-person telemetry in permitted areas",
  },
  {
    key: "multiple_people_moving",
    name: "Multiple people moving",
    group: "Operations",
    detail: "Simultaneous sustained movement",
  },
  {
    key: "fall",
    name: "Person down",
    group: "Safety / HSE",
    detail: "Experimental",
  },
] as const;
