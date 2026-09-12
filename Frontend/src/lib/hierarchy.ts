import type { Camera, Hierarchy } from "./types";

export type LocationSelection =
  | Readonly<{ kind: "all" }>
  | Readonly<{ kind: "unassigned" }>
  | Readonly<{ kind: "id"; id: string }>;

export const ALL_LOCATIONS: LocationSelection = Object.freeze({ kind: "all" });
export const UNASSIGNED_BRANCH: LocationSelection = Object.freeze({
  kind: "unassigned",
});

export interface LocationOption {
  selection: LocationSelection;
  name: string;
  count: number;
}

export interface WallFilterPreference {
  branch: LocationSelection;
  area: LocationSelection;
}

export function locationIdSelection(id: string): LocationSelection {
  return { kind: "id", id };
}

export function encodeLocationSelection(selection: LocationSelection): string {
  if (selection.kind === "id") return `id:${encodeURIComponent(selection.id)}`;
  return selection.kind;
}

export function decodeLocationSelection(
  value: string,
): LocationSelection | undefined {
  if (value === "all") return ALL_LOCATIONS;
  if (value === "unassigned") return UNASSIGNED_BRANCH;
  if (!value.startsWith("id:")) return undefined;
  try {
    return locationIdSelection(decodeURIComponent(value.slice(3)));
  } catch {
    return undefined;
  }
}

export function sameLocationSelection(
  left: LocationSelection,
  right: LocationSelection,
): boolean {
  return (
    left.kind === right.kind &&
    (left.kind !== "id" || (right.kind === "id" && left.id === right.id))
  );
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
  branch: LocationSelection,
  area: LocationSelection,
  query: string,
): Camera[] {
  const areas = locationMaps(hierarchy);
  const needle = query.trim().toLowerCase();
  return cameras.filter((camera) => {
    const location = camera.area_id ? areas.get(camera.area_id) : undefined;
    if (branch.kind === "id" && location?.branchId !== branch.id) return false;
    if (branch.kind === "unassigned" && location) return false;
    if (area.kind === "id" && camera.area_id !== area.id) return false;
    if (area.kind === "unassigned" && location) return false;
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
  const configured = hierarchy.branches.map((branch) => {
    const selection = locationIdSelection(branch.id);
    return {
      selection,
      name: branch.name,
      count: filterCameras(cameras, hierarchy, selection, ALL_LOCATIONS, "")
        .length,
    };
  });
  const unassigned = filterCameras(
    cameras,
    hierarchy,
    UNASSIGNED_BRANCH,
    ALL_LOCATIONS,
    "",
  ).length;
  return [
    { selection: ALL_LOCATIONS, name: "All branches", count: cameras.length },
    ...configured,
    {
      selection: UNASSIGNED_BRANCH,
      name: "Unassigned",
      count: unassigned,
    },
  ];
}

export function areaOptions(
  cameras: Camera[],
  hierarchy: Hierarchy,
  branch: LocationSelection,
): LocationOption[] {
  const branches =
    branch.kind === "all"
      ? hierarchy.branches
      : branch.kind === "id"
        ? hierarchy.branches.filter((item) => item.id === branch.id)
        : [];
  const count = filterCameras(
    cameras,
    hierarchy,
    branch,
    ALL_LOCATIONS,
    "",
  ).length;
  return [
    { selection: ALL_LOCATIONS, name: "All areas", count },
    ...branches.flatMap((item) =>
      item.areas.map((area) => {
        const selection = locationIdSelection(area.id);
        return {
          selection,
          name: area.name,
          count: filterCameras(cameras, hierarchy, branch, selection, "")
            .length,
        };
      }),
    ),
  ];
}

export function reconcileAreaSelection(
  hierarchy: Hierarchy,
  branch: LocationSelection,
  area: LocationSelection,
): LocationSelection {
  if (area.kind === "all") return area;
  if (branch.kind === "unassigned" || area.kind !== "id") return ALL_LOCATIONS;
  const valid = hierarchy.branches.some(
    (item) =>
      (branch.kind === "all" || item.id === branch.id) &&
      item.areas.some((candidate) => candidate.id === area.id),
  );
  return valid ? area : ALL_LOCATIONS;
}

export function reconcileBranchSelection(
  hierarchy: Hierarchy,
  branch: LocationSelection,
): LocationSelection {
  if (branch.kind !== "id") return branch;
  return hierarchy.branches.some((item) => item.id === branch.id)
    ? branch
    : ALL_LOCATIONS;
}

export function hierarchyPreferenceKey(username: string): string {
  return `argus.wall.filters.v1:${encodeURIComponent(username)}`;
}

function storedSelection(value: unknown): LocationSelection | undefined {
  if (value && typeof value === "object") {
    const selection = value as { kind?: unknown; id?: unknown };
    if (selection.kind === "all") return ALL_LOCATIONS;
    if (selection.kind === "unassigned") return UNASSIGNED_BRANCH;
    if (selection.kind === "id" && typeof selection.id === "string")
      return locationIdSelection(selection.id);
    return undefined;
  }
  if (typeof value !== "string") return undefined;
  if (value === "all" || value === "virtual:all") return ALL_LOCATIONS;
  if (value === "unassigned" || value === "virtual:unassigned")
    return UNASSIGNED_BRANCH;
  if (value.startsWith("id:")) {
    try {
      return locationIdSelection(decodeURIComponent(value.slice(3)));
    } catch {
      return undefined;
    }
  }
  return locationIdSelection(value);
}

export function loadWallFilterPreference(
  storage: Pick<Storage, "getItem" | "setItem">,
  username: string,
): WallFilterPreference {
  try {
    const parsed = JSON.parse(
      storage.getItem(hierarchyPreferenceKey(username)) || "null",
    );
    const branch = storedSelection(parsed?.branch ?? parsed?.branchId);
    const area = storedSelection(parsed?.area ?? parsed?.areaId);
    if (branch && area) return { branch, area };
  } catch {
    /* Invalid preferences fall back to the complete wall. */
  }
  return { branch: ALL_LOCATIONS, area: ALL_LOCATIONS };
}

export function saveWallFilterPreference(
  storage: Pick<Storage, "getItem" | "setItem">,
  username: string,
  preference: WallFilterPreference,
): void {
  storage.setItem(hierarchyPreferenceKey(username), JSON.stringify(preference));
}
