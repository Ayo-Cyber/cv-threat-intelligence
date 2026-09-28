import type { GateStatus, PullProgress } from "../components/VerifierDownload";

/** The setup wizard's order, and what gates Finish.
 *
 * The on-device AI model comes FIRST: scene mapping (which cameras see what)
 * and alert verification both run on it, so a site finished without it has
 * cameras whose scenes were never read and alerts nobody checked. A Skip
 * button here produced exactly that ("agent mapping wont run", 28 Sep). So
 * there is no skip -- but the download does not block the rest of setup
 * either: it runs while cameras are added, and only Finish waits for it. */
export const SETUP_STEPS = [
  "AI model",
  "Locations",
  "Cameras",
  "Detectors",
  "Alerts",
  "Finish",
] as const;

export type SetupStep = (typeof SETUP_STEPS)[number];

export function stepIndex(step: SetupStep): number {
  return SETUP_STEPS.indexOf(step);
}

/** Whether the on-device AI is usable, from what the engine reports. */
export function modelReady(gate: GateStatus | null): boolean {
  return gate?.mode === "live";
}

/** Why Finish is disabled, in the operator's words, or null when it is not.
 * Demo mode has no model to wait for. */
export function finishBlockedReason(input: {
  mode: "demo" | "engine";
  cameras: number;
  gate: GateStatus | null;
  pull: PullProgress | null;
  online?: boolean;
}): string | null {
  if (input.cameras === 0) return "Add at least one camera first.";
  if (input.mode === "demo") return null;
  if (modelReady(input.gate)) return null;
  if (input.pull?.state === "pulling") {
    const percent = Math.max(0, Math.min(100, Math.round(input.pull.percent ?? 0)));
    return `Finish unlocks when the on-device AI finishes downloading (${percent}%). Keep this window open; you can carry on configuring.`;
  }
  if (input.online === false)
    return "The on-device AI model has not been downloaded and this computer is offline. Connect to the internet and the download resumes.";
  if (input.pull?.state === "error")
    return `The AI model download stopped: ${input.pull.detail || "unknown error"}. Retry it on the AI model step; Finish unlocks when it completes.`;
  if (!input.gate) return "Waiting for the engine to report the AI model's status.";
  return "The on-device AI model is not ready yet. Finish unlocks when it is -- go back to the AI model step to download it.";
}
