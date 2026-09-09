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
  mapping: { status: string; provenance?: string; reviewed_by?: string };
}
export interface Camera {
  id: string;
  source: string;
  area_id?: string;
  area?: string;
  config?: string;
  custom_rules?: { question: string; dwell: number }[];
  custom_threats?: { name: string; description: string }[];
  demo_video?: string;
  snapshot?: string;
  [key: string]: any;
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
  site: Json;
  monitor: Json;
  english: Json;
  auth: Auth;
}
export interface Transport {
  invoke<T = any>(method: string, args?: unknown[]): Promise<T>;
}
declare global {
  interface Window {
    argusDesktop?: {
      invoke<T = any>(method: string, args?: unknown[]): Promise<T>;
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
    key: "fall",
    name: "Person down",
    group: "Safety / HSE",
    detail: "Experimental",
  },
] as const;
