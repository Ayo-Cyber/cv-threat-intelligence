import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import IncidentDetails from "../src/components/IncidentDetails";
import type { Incident, Transport } from "../src/lib/types";

it("shows review guidance without rendering internal concealment diagnostics", () => {
  const reason = "NEEDS REVIEW: The model cites a single frame rather than an action sequence. The hiding location is not established.";
  const event = {
    id: "test-incident", camera_id: "retail_1", rule: "product_concealment",
    reason, priority: "high", ts: 1, triage_state: "new",
  } as Incident;
  const html = renderToStaticMarkup(createElement(IncidentDetails, {
    event, api: {} as Transport, onChange: async () => {},
  }));
  expect(html).toContain("Argus has flagged possible product concealment. Please review the recorded clip.");
  expect(html).toContain("Review required");
  expect(html).not.toContain("single frame");
  expect(html).not.toContain("hiding location");
  expect(html).not.toContain("Assessment details");
  expect(event.reason).toBe(reason);
});
