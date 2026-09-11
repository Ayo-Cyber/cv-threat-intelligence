import type { Camera, Hierarchy } from "./types";

export const ALL_LOCATIONS = "virtual:all";
export const UNASSIGNED_BRANCH = "virtual:unassigned";
export const ASSIGN_LATER = "virtual:assign-later";
const ID_TOKEN_PREFIX = "id:";

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

export function locationIdToken(id: string): string {
  return `${ID_TOKEN_PREFIX}${encodeURIComponent(id)}`;
}

export function decodeLocationId(token: string): string | undefined {
  if (!token.startsWith(ID_TOKEN_PREFIX)) return undefined;
  try {
    return decodeURIComponent(token.slice(ID_TOKEN_PREFIX.length));
  } catch {
    return undefined;
  }
}

function selectedBackendId(token: string): string | undefined {
  const decoded = decodeLocationId(token);
  if (decoded !== undefined) return decoded;
  if (token.startsWith("virtual:")) return undefined;
  return token;
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
  const selectedBranchId = selectedBackendId(branchId);
  const selectedAreaId = selectedBackendId(areaId);
  return cameras.filter((camera) => {
    const location = camera.area_id ? areas.get(camera.area_id) : undefined;
    const cameraBranch = location?.branchId ?? UNASSIGNED_BRANCH;
    if (
      branchId === UNASSIGNED_BRANCH
        ? cameraBranch !== UNASSIGNED_BRANCH
        : branchId !== ALL_LOCATIONS && cameraBranch !== selectedBranchId
    )
      return false;
    if (areaId !== ALL_LOCATIONS && camera.area_id !== selectedAreaId)
      return false;
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
    id: locationIdToken(branch.id),
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
      : hierarchy.branches.filter(
          (branch) => branch.id === selectedBackendId(branchId),
        );
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
        id: locationIdToken(area.id),
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
  const selectedBranchId = selectedBackendId(branchId);
  const selectedAreaId = selectedBackendId(areaId);
  const valid = hierarchy.branches.some(
    (branch) =>
      (branchId === ALL_LOCATIONS || branch.id === selectedBranchId) &&
      branch.areas.some((area) => area.id === selectedAreaId),
  );
  return valid ? areaId : ALL_LOCATIONS;
}

export function reconcileBranchSelection(
  hierarchy: Hierarchy,
  branchId: string,
): string {
  if (branchId === ALL_LOCATIONS || branchId === UNASSIGNED_BRANCH)
    return branchId;
  const selectedBranchId = selectedBackendId(branchId);
  return hierarchy.branches.some((branch) => branch.id === selectedBranchId)
    ? branchId
    : ALL_LOCATIONS;
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
      return {
        branchId:
          parsed.branchId === "all"
            ? ALL_LOCATIONS
            : parsed.branchId === "unassigned"
              ? UNASSIGNED_BRANCH
              : parsed.branchId.startsWith("virtual:") ||
                  decodeLocationId(parsed.branchId) !== undefined
                ? parsed.branchId
                : locationIdToken(parsed.branchId),
        areaId:
          parsed.areaId === "all"
            ? ALL_LOCATIONS
            : parsed.areaId.startsWith("virtual:") ||
                decodeLocationId(parsed.areaId) !== undefined
              ? parsed.areaId
              : locationIdToken(parsed.areaId),
      };
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
