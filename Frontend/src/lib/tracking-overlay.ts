export type TrackingPreference = "global" | "show" | "hide";

export type TrackingOverrides = Readonly<Record<string, TrackingPreference>>;

type PreferenceStorage = Pick<Storage, "getItem" | "setItem">;

function trackingPreferenceKey(operatorId: string): string {
  return `argus:tracking-overlay:${operatorId}`;
}

export function loadGlobalTrackingPreference(
  storage: PreferenceStorage,
  operatorId: string,
): boolean {
  return storage.getItem(trackingPreferenceKey(operatorId)) === "true";
}

export function saveGlobalTrackingPreference(
  storage: PreferenceStorage,
  operatorId: string,
  visible: boolean,
): void {
  storage.setItem(trackingPreferenceKey(operatorId), String(visible));
}

export function cameraTrackingPreference(
  overrides: TrackingOverrides,
  cameraId: string,
): TrackingPreference {
  return overrides[cameraId] ?? "global";
}

export function updateTrackingOverride(
  overrides: TrackingOverrides,
  cameraId: string,
  preference: TrackingPreference,
): TrackingOverrides {
  const next = { ...overrides };
  if (preference === "global") delete next[cameraId];
  else next[cameraId] = preference;
  return next;
}

export function trackingVisible(
  globalVisible: boolean,
  cameraOverride: TrackingPreference,
): boolean {
  if (cameraOverride === "show") return true;
  if (cameraOverride === "hide") return false;
  return globalVisible;
}
