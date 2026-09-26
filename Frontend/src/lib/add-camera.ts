/**
 * What is still missing before a camera can be added.
 *
 * "Add camera" is disabled on !id || !source || !placementReady and said
 * nothing about which. On 26 Sep an operator had a reachable webcam, a
 * branch and an area chosen, and a dead button — because the Camera name
 * was empty and its placeholder ("reception_01") reads as a filled-in
 * value. This is the same shape as the pilot's "Finish setup wasn't
 * clicking" (#172): a control that refuses without saying why.
 */
export function missingToAddCamera(fields: {
  id: string;
  source: string;
  placementReady: boolean;
}): string[] {
  const missing: string[] = [];
  if (!fields.id.trim()) missing.push("a camera name");
  if (!fields.source.trim())
    missing.push("a camera source (0 for a webcam, or an RTSP address)");
  if (!fields.placementReady) missing.push("a branch and an area");
  return missing;
}

/** One sentence naming everything still needed, or null when ready. */
export function addCameraBlockedReason(fields: {
  id: string;
  source: string;
  placementReady: boolean;
}): string | null {
  const missing = missingToAddCamera(fields);
  if (!missing.length) return null;
  const list =
    missing.length === 1
      ? missing[0]
      : `${missing.slice(0, -1).join(", ")} and ${missing[missing.length - 1]}`;
  return `Still needed before this camera can be added: ${list}.`;
}
