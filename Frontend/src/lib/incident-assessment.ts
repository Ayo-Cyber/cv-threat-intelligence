import type { Incident } from "./types";

const CONCEALMENT_SUMMARY = "Argus has flagged possible product concealment. Please review the recorded clip.";

export function incidentAssessment(event: Pick<Incident, "rule" | "reason">) {
  const original = event.reason || "No assessment supplied by the engine.";
  const concealment = event.rule.toLowerCase().includes("concealment");
  const reviewMessage = concealment && (
    original.startsWith("NEEDS REVIEW:") ||
    original.includes("Possible product concealment") ||
    original.includes("possible product concealment")
  );
  // Legacy records retain their original assessment for audit. Only replace
  // anatomically specific concealment claims, never failure/rejection messages.
  const uncertainDestination = event.rule.toLowerCase().includes("concealment") &&
    !original.startsWith("Visual assessment (AI): ") &&
    /\bpockets?\b/i.test(original) &&
    !/unverified|mock provider|insufficient|rejected|below confidence|no (?:visible |clear )?insertion/i.test(original);
  return {
    summary: reviewMessage || uncertainDestination ? CONCEALMENT_SUMMARY : original,
    original: reviewMessage || uncertainDestination ? original : null,
  };
}
