import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import ObjectWatchlistManager from "../src/components/ObjectWatchlistManager";
import type { Auth, Hierarchy, ObjectTarget, Transport } from "../src/lib/types";

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
  examples: [],
  negative_examples: [],
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
        initialTargets: [target],
      }),
    );

    expect(html).toContain("Object watchlists");
    expect(html).toContain("Create target");
    expect(html).toContain("Upload example");
    expect(html).toContain("Activate");
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
    expect(html).not.toContain("Create target");
    expect(html).not.toContain("Upload example");
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
