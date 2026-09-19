import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import ObjectWatchlistManager, {
  objectCategories,
} from "../src/components/ObjectWatchlistManager";
import type {
  Auth,
  Hierarchy,
  ObjectTarget,
  Transport,
} from "../src/lib/types";

const hierarchy: Hierarchy = {
  organization: { id: "org", name: "Chi" },
  branches: [
    {
      id: "branch",
      name: "Lagos Factory",
      areas: [
        { id: "storage", name: "Storage", branch_id: "branch", cameras: [] },
        {
          id: "loading_bay",
          name: "Loading Bay",
          branch_id: "branch",
          cameras: [],
        },
      ],
    },
  ],
  unassigned_cameras: [],
};

const target: ObjectTarget = {
  id: "chi-carton",
  label: "Chi carton",
  category: "product",
  aliases: ["milk carton"],
  review_state: "draft",
  min_similarity: 0.72,
  allowed_zone_ids: ["storage"],
  grounding_description: "sealed drink carton with a printed front panel",
  examples: [],
  negative_examples: [],
  can_activate: false,
  reasons: ["no_reviewed_positive_examples"],
};

const api: Transport = {
  invoke: vi.fn(async () => ({ targets: [target] })),
};

const owner: Auth = {
  configured: true,
  signed_in: true,
  username: "ayo",
  role: "owner",
  permissions: ["view_live", "configure_cameras"],
};

const installer: Auth = {
  ...owner,
  role: "installer",
  permissions: ["view_live", "configure_cameras", "configure_detectors"],
};

const operator: Auth = {
  ...owner,
  role: "operator",
  permissions: ["view_live"],
};

describe("ObjectWatchlistManager", () => {
  it("shows enrollment controls for owners with camera configuration permission", () => {
    const html = renderToStaticMarkup(
      React.createElement(ObjectWatchlistManager, {
        transport: api,
        authState: owner,
        hierarchy,
        initialTargets: [],
        initialRuntime: {
          status: "structurally_available",
          backend: "siglip",
          fingerprint: "siglip-local",
          structurally_available: true,
          executable_verified: false,
          reason_codes: [],
        },
      }),
    );

    expect(html).toContain("Object watchlists");
    expect(html).toContain("Add the object photo");
    expect(html).toContain("Choose photo");
    expect(html).toContain("Optional naming details");
    expect(html).toContain("Visual description");
    expect(html).toContain("Prepare recognition");
    expect(html).toContain("Configured");
    expect(html).toContain("prepare examples to verify execution");
    expect(html).not.toContain("Allowed zone ids");
    expect(objectCategories).toEqual([
      "product",
      "vehicle",
      "pallet",
      "ppe",
      "custom",
    ]);
    expect(objectCategories).not.toContain("equipment");
    expect(objectCategories).not.toContain("other");
  });

  it("keeps recognition activation separate from object_seen alert rules", () => {
    const html = renderToStaticMarkup(
      React.createElement(ObjectWatchlistManager, {
        transport: api,
        authState: installer,
        hierarchy,
        cameras: [{ id: "cam-1", source: "rtsp://camera" }],
        initialTargets: [{ ...target, can_activate: true }],
        initialRuntime: {
          status: "ready",
          backend: "siglip",
          fingerprint: "siglip-local",
          reason_codes: [],
        },
      }),
    );

    expect(html).toContain("Recognize this object");
    expect(html).toContain("Camera for alert rule");
    expect(html).toContain("Enable object_seen alert");
  });

  it("explains unavailable local model runtime without hash fallback", () => {
    const html = renderToStaticMarkup(
      React.createElement(ObjectWatchlistManager, {
        transport: api,
        authState: owner,
        hierarchy,
        initialTargets: [target],
        initialRuntime: {
          status: "unavailable",
          backend: "siglip",
          fingerprint: null,
          reason_codes: ["missing_local_model"],
        },
      }),
    );

    expect(html).toContain("Unavailable");
    expect(html).toContain("Choose the local semantic model files below.");
    expect(html).not.toContain("hash");
  });

  it("renders targets read-only for operators", () => {
    const html = renderToStaticMarkup(
      React.createElement(ObjectWatchlistManager, {
        transport: api,
        authState: operator,
        hierarchy,
        initialTargets: [{ ...target, review_state: "active" }],
      }),
    );

    expect(html).toContain("Chi carton");
    expect(html).toContain("Read-only");
    expect(html).not.toContain("Choose photo");
    expect(html).not.toContain("Enable object_seen alert");
  });

  it("labels demo object targets as fixture-backed instead of inferred", () => {
    const html = renderToStaticMarkup(
      React.createElement(ObjectWatchlistManager, {
        transport: api,
        authState: owner,
        hierarchy,
        mode: "demo",
        initialTargets: [target],
      }),
    );

    expect(html).toContain("fixture-backed");
    expect(html).toContain("No local inference runs in demo mode");
  });
});
