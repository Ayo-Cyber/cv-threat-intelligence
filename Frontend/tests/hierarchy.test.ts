import { describe, expect, it } from "vitest";
import {
  ALL_LOCATIONS,
  UNASSIGNED_BRANCH,
  areaOptions,
  branchOptions,
  filterCameras,
  hierarchyPreferenceKey,
  loadWallFilterPreference,
  reconcileAreaSelection,
  saveWallFilterPreference,
} from "../src/lib/hierarchy";
import type { Camera, Hierarchy } from "../src/lib/types";
import { createDemo, initialDemo } from "../src/lib/demo";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import LocationManager from "../src/components/LocationManager";
import AddCamera from "../src/components/AddCamera";
import type { Auth, Transport } from "../src/lib/types";

const hierarchy: Hierarchy = {
  organization: { id: "org-1", name: "Acme" },
  branches: [
    {
      id: "ikeja",
      name: "Ikeja Outlet",
      areas: [
        {
          id: "checkout",
          name: "Checkout",
          branch_id: "ikeja",
          cameras: [],
        },
      ],
    },
    {
      id: "lekki",
      name: "Lekki Outlet",
      areas: [
        {
          id: "stockroom",
          name: "Stockroom",
          branch_id: "lekki",
          cameras: [],
        },
      ],
    },
  ],
  unassigned_cameras: [],
};

const cameras: Camera[] = [
  {
    id: "checkout-1",
    source: "one",
    area_id: "checkout",
    branch_id: "ikeja",
  },
  {
    id: "stock-1",
    source: "two",
    area_id: "stockroom",
    branch_id: "lekki",
  },
  { id: "needs-placement", source: "three" },
  { id: "stale-area", source: "four", area_id: "missing" },
];

describe("hierarchy selectors", () => {
  it("filters organization then branch then area then query", () => {
    expect(
      filterCameras(cameras, hierarchy, "ikeja", "checkout", "CHECKOUT"),
    ).toEqual([cameras[0]]);
    expect(
      filterCameras(cameras, hierarchy, "lekki", ALL_LOCATIONS, "stockroom"),
    ).toEqual([cameras[1]]);
  });

  it("keeps incomplete cameras visible only in the unassigned group", () => {
    expect(
      filterCameras(
        cameras,
        hierarchy,
        UNASSIGNED_BRANCH,
        ALL_LOCATIONS,
        "",
      ).map((camera) => camera.id),
    ).toEqual(["needs-placement", "stale-area"]);
    expect(
      filterCameras(cameras, hierarchy, ALL_LOCATIONS, ALL_LOCATIONS, "").map(
        (camera) => camera.id,
      ),
    ).toEqual(["checkout-1", "stock-1", "needs-placement", "stale-area"]);
  });

  it("uses IDs rather than legacy labels to place cameras", () => {
    const mislabeled = {
      id: "label-only",
      source: "five",
      area: "Checkout",
    };
    expect(
      filterCameras([mislabeled], hierarchy, "ikeja", ALL_LOCATIONS, ""),
    ).toEqual([]);
    expect(
      filterCameras(
        [mislabeled],
        hierarchy,
        UNASSIGNED_BRANCH,
        ALL_LOCATIONS,
        "",
      ),
    ).toEqual([mislabeled]);
  });

  it("orders configured branches first and virtual Unassigned last", () => {
    expect(branchOptions(cameras, hierarchy)).toEqual([
      { id: ALL_LOCATIONS, name: "All branches", count: 4 },
      { id: "ikeja", name: "Ikeja Outlet", count: 1 },
      { id: "lekki", name: "Lekki Outlet", count: 1 },
      { id: UNASSIGNED_BRANCH, name: "Unassigned", count: 2 },
    ]);
  });

  it("returns only areas in the selected branch with live counts", () => {
    expect(areaOptions(cameras, hierarchy, "ikeja")).toEqual([
      { id: ALL_LOCATIONS, name: "All areas", count: 1 },
      { id: "checkout", name: "Checkout", count: 1 },
    ]);
    expect(areaOptions(cameras, hierarchy, UNASSIGNED_BRANCH)).toEqual([
      { id: ALL_LOCATIONS, name: "All areas", count: 2 },
    ]);
  });

  it("clears an area selection that is outside the selected branch", () => {
    expect(reconcileAreaSelection(hierarchy, "ikeja", "stockroom")).toBe(
      ALL_LOCATIONS,
    );
    expect(reconcileAreaSelection(hierarchy, "ikeja", "checkout")).toBe(
      "checkout",
    );
  });

  it("keys wall preferences per signed-in username", () => {
    expect(hierarchyPreferenceKey("Demi O.")).toBe(
      "argus.wall.filters.v1:Demi%20O.",
    );
    expect(hierarchyPreferenceKey("ayo")).not.toBe(
      hierarchyPreferenceKey("Demi O."),
    );
  });

  it("round-trips separate wall filters for separate usernames", () => {
    const values = new Map<string, string>();
    const storage = {
      getItem: (key: string) => values.get(key) || null,
      setItem: (key: string, value: string) => values.set(key, value),
    };
    saveWallFilterPreference(storage, "ayo", {
      branchId: "ikeja",
      areaId: "checkout",
    });
    saveWallFilterPreference(storage, "demi", {
      branchId: "lekki",
      areaId: "stockroom",
    });
    expect(loadWallFilterPreference(storage, "ayo")).toEqual({
      branchId: "ikeja",
      areaId: "checkout",
    });
    expect(loadWallFilterPreference(storage, "demi")).toEqual({
      branchId: "lekki",
      areaId: "stockroom",
    });
  });

  it("falls back safely when a saved wall preference is malformed", () => {
    expect(
      loadWallFilterPreference(
        { getItem: () => "not-json", setItem: () => {} },
        "ayo",
      ),
    ).toEqual({ branchId: ALL_LOCATIONS, areaId: ALL_LOCATIONS });
  });
});

describe("demo hierarchy", () => {
  it("provides a representative organization with two configured branches", () => {
    const state = initialDemo();
    expect(state.organization.name).toBe("Deluxe Paints Nigeria");
    expect(state.branches).toHaveLength(2);
    expect(state.areas.every((area) => area.branch_id)).toBe(true);
    expect(state.cameras.every((camera) => camera.branch_id)).toBe(true);
  });

  it("serves hierarchy reads from isolated demo state", async () => {
    const writes: string[] = [];
    const api = createDemo({
      getItem: () => null,
      setItem: (key) => writes.push(key),
    });
    const organization = await api.invoke("organization");
    const branches = await api.invoke("list_branches");
    const tree = await api.invoke<Hierarchy>("hierarchy");

    expect(tree.organization).toEqual(organization);
    expect(tree.branches.map((branch) => branch.id)).toEqual(
      branches.map((branch: { id: string }) => branch.id),
    );
    expect(
      filterCameras(
        await api.invoke<Camera[]>("list_cameras"),
        tree,
        tree.branches[0].id,
        ALL_LOCATIONS,
        "",
      ).length,
    ).toBeGreaterThan(0);
    expect(writes).toEqual([]);
  });

  it("upgrades saved pre-hierarchy demo locations without hiding cameras", async () => {
    const saved = initialDemo() as any;
    delete saved.organization;
    delete saved.branches;
    saved.areas.forEach(
      (area: { branch_id?: string }) => delete area.branch_id,
    );
    saved.cameras.forEach((camera: Camera) => delete camera.branch_id);
    const api = createDemo({
      getItem: () => JSON.stringify(saved),
      setItem: () => {},
    });
    const tree = await api.invoke<Hierarchy>("hierarchy");
    const nested = tree.branches.flatMap((branch) =>
      branch.areas.flatMap((area) => area.cameras),
    );
    expect([...nested, ...tree.unassigned_cameras]).toHaveLength(4);
  });
});

describe("location management permissions", () => {
  const api: Transport = { invoke: async () => ({ ok: true }) };
  const managedHierarchy: Hierarchy = {
    ...hierarchy,
    branches: [
      ...hierarchy.branches,
      { id: "empty", name: "Empty Branch", areas: [] },
    ],
  };
  const auth = (permissions: string[]): Auth => ({
    configured: true,
    signed_in: true,
    username: "user",
    role: "custom",
    permissions,
  });
  const renderManager = (permissions: string[], mode: "demo" | "engine") =>
    renderToStaticMarkup(
      createElement(LocationManager, {
        api,
        mode,
        auth: auth(permissions),
        hierarchy: managedHierarchy,
        cameras,
        onChange: async () => {},
        notify: () => {},
      }),
    );

  it("shows operators hierarchy labels without mutation controls", () => {
    const markup = renderManager([], "engine");
    expect(markup).toContain("Acme");
    expect(markup).toContain("Ikeja Outlet");
    expect(markup).toContain("Checkout");
    expect(markup).not.toContain("Save organization");
    expect(markup).not.toContain("Add branch");
    expect(markup).not.toContain("Assign camera");
  });

  it("separates organization and camera configuration permissions", () => {
    const siteMarkup = renderManager(["configure_site"], "engine");
    expect(siteMarkup).toContain("Save organization");
    expect(siteMarkup).not.toContain("Add branch");

    const cameraMarkup = renderManager(["configure_cameras"], "engine");
    expect(cameraMarkup).not.toContain("Save organization");
    expect(cameraMarkup).toContain("Add branch");
    expect(cameraMarkup).toContain("Assign camera");
  });

  it("disables deletion for non-empty branches", () => {
    const markup = renderManager(["configure_cameras"], "engine");
    expect(markup).toMatch(/title="Delete Ikeja Outlet"[^>]*disabled/);
    expect(markup).toMatch(/title="Delete Empty Branch"(?![^>]*disabled)/);
  });

  it("keeps demo hierarchy controls read-only", () => {
    const markup = renderManager(
      ["configure_site", "configure_cameras"],
      "demo",
    );
    expect(markup).toContain("Sample location hierarchy");
    expect(markup).not.toContain("Save organization");
    expect(markup).not.toContain("Add branch");
  });
});

describe("camera onboarding hierarchy", () => {
  it("asks for a branch before offering a scoped area or Assign later", () => {
    const markup = renderToStaticMarkup(
      createElement(AddCamera, {
        api: { invoke: async () => ({ ok: true }) },
        mode: "engine",
        areas: [],
        hierarchy,
        onAdded: async () => {},
      } as any),
    );
    expect(markup).toContain("Select a branch");
    expect(markup).toContain("Assign later");
    expect(markup).not.toContain("Select an area");
  });
});
