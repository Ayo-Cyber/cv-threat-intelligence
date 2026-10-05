import { expect, it } from "vitest";
import { incidentAssessment } from "../src/lib/incident-assessment";

it("uses client-facing copy and retains structured observations as details", () => {
  const reason = "Visual assessment (AI): The person appears to move an item into a pocket. Possible product concealment; review the recorded sequence.";
  expect(incidentAssessment({ rule: "product_concealment", reason })).toEqual({
    summary: "Argus has flagged possible product concealment. Please review the recorded clip.", original: reason,
  });
});

it("qualifies legacy pocket claims without rewriting the original or guessing waist", () => {
  const reason = "The woman is shown reaching into her pocket and appearing to place an item inside.";
  const result = incidentAssessment({ rule: "product_concealment", reason });
  expect(result.summary).toContain("Argus has flagged possible product concealment");
  expect(result.summary).not.toMatch(/pocket|waist/);
  expect(result.original).toBe(reason);
});

it("keeps review diagnostics out of the customer summary without discarding them", () => {
  const reason = "NEEDS REVIEW: The model cites a single frame.";
  const result = incidentAssessment({ rule: "product_concealment", reason });
  expect(result.summary).toBe("Argus has flagged possible product concealment. Please review the recorded clip.");
  expect(result.original).toBe(reason);
});

it("preserves failures, negative assessments and unrelated rules", () => {
  for (const reason of ["UNVERIFIED: pocket action not checked", "Mock provider: pocket test",
    "Insufficient evidence of pocket insertion", "No visible insertion into a pocket"]) {
    expect(incidentAssessment({ rule: "product_concealment", reason }).summary).toBe(reason);
  }
  expect(incidentAssessment({ rule: "custom_rule", reason: "A pocket is visible" }).original).toBeNull();
});
