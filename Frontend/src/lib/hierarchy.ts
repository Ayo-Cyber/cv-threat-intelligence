import type { Camera, Hierarchy } from "./types";

export const ALL_LOCATIONS = "all";
export const UNASSIGNED_BRANCH = "unassigned";

export interface LocationOption {
  id: string;
  name: string;
  count: number;
}

export interface WallFilterPreference {
  branchId: string;
  areaId: string;
}

function locationMaps(hierarchy: Hierarchy) {
  const areas = new Map<
    string,
    { branchId: string; branchName: string; areaName: string }
  >();
  for (const branch of hierarchy.branches) {
    for (const area of branch.areas) {
      areas.set(area.id, {
        branchId: branch.id,
        branchName: branch.name,
        areaName: area.name,
      });
    }
  }
  return areas;
}

export function filterCameras(
  cameras: Camera[],
  hierarchy: Hierarchy,
  branchId: string,
  areaId: string,
  query: string,
): Camera[] {
  const areas = locationMaps(hierarchy);
  const needle = query.trim().toLowerCase();
  return cameras.filter((camera) => {
    const location = camera.area_id ? areas.get(camera.area_id) : undefined;
    const cameraBranch = location?.branchId ?? UNASSIGNED_BRANCH;
    if (branchId !== ALL_LOCATIONS && cameraBranch !== branchId) return false;
    if (areaId !== ALL_LOCATIONS && camera.area_id !== areaId) return false;
    if (!needle) return true;
    return `${camera.id} ${location?.branchName ?? "Unassigned"} ${location?.areaName ?? ""}`
      .toLowerCase()
      .includes(needle);
  });
}

export function branchOptions(
  cameras: Camera[],
  hierarchy: Hierarchy,
): LocationOption[] {
  const all = cameras.length;
  const configured = hierarchy.branches.map((branch) => ({
    id: branch.id,
    name: branch.name,
    count: filterCameras(cameras, hierarchy, branch.id, ALL_LOCATIONS, "")
      .length,
  }));
  const unassigned = filterCameras(
    cameras,
    hierarchy,
    UNASSIGNED_BRANCH,
    ALL_LOCATIONS,
    "",
  ).length;
  return [
    { id: ALL_LOCATIONS, name: "All branches", count: all },
    ...configured,
    { id: UNASSIGNED_BRANCH, name: "Unassigned", count: unassigned },
  ];
}

export function areaOptions(
  cameras: Camera[],
  hierarchy: Hierarchy,
  branchId: string,
): LocationOption[] {
  const branches =
    branchId === ALL_LOCATIONS
      ? hierarchy.branches
      : hierarchy.branches.filter((branch) => branch.id === branchId);
  const count = filterCameras(
    cameras,
    hierarchy,
    branchId,
    ALL_LOCATIONS,
    "",
  ).length;
  return [
    { id: ALL_LOCATIONS, name: "All areas", count },
    ...branches.flatMap((branch) =>
      branch.areas.map((area) => ({
        id: area.id,
        name: area.name,
        count: filterCameras(cameras, hierarchy, branchId, area.id, "").length,
      })),
    ),
  ];
}

export function reconcileAreaSelection(
  hierarchy: Hierarchy,
  branchId: string,
  areaId: string,
): string {
  if (areaId === ALL_LOCATIONS) return areaId;
  if (branchId === UNASSIGNED_BRANCH) return ALL_LOCATIONS;
  const valid = hierarchy.branches.some(
    (branch) =>
      (branchId === ALL_LOCATIONS || branch.id === branchId) &&
      branch.areas.some((area) => area.id === areaId),
  );
  return valid ? areaId : ALL_LOCATIONS;
}

export function hierarchyPreferenceKey(username: string): string {
  return `argus.wall.filters.v1:${encodeURIComponent(username)}`;
}

export function loadWallFilterPreference(
  storage: Pick<Storage, "getItem" | "setItem">,
  username: string,
): WallFilterPreference {
  try {
    const parsed = JSON.parse(
      storage.getItem(hierarchyPreferenceKey(username)) || "null",
    );
    if (
      typeof parsed?.branchId === "string" &&
      typeof parsed?.areaId === "string"
    )
      return { branchId: parsed.branchId, areaId: parsed.areaId };
  } catch {
    /* Invalid preferences fall back to the complete wall. */
  }
  return { branchId: ALL_LOCATIONS, areaId: ALL_LOCATIONS };
}

export function saveWallFilterPreference(
  storage: Pick<Storage, "getItem" | "setItem">,
  username: string,
  preference: WallFilterPreference,
): void {
  storage.setItem(hierarchyPreferenceKey(username), JSON.stringify(preference));
}
