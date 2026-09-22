// What the incident card shows for an alert's evidence, decided in one place.
//
// Three things used to collapse into "No recorded evidence is available":
//  * the clip is still being written (a critical alert's row exists ~20s
//    before its evidence does, and the push loop hands us that early row);
//  * the clip exists but this device cannot decode it (OpenCV's mp4v on a
//    Windows build), which Chromium renders as a silent black box;
//  * there genuinely is nothing on disk.
// The frames written next to the clip play as a cine-loop in the second case.

export interface ClipReply {
  uri?: string | null;
  frames?: string[];
  subject?: string | null;
  fps?: number | null;
  codec?: string | null;
  pending?: boolean;
}

export type EvidenceView = "video" | "frames" | "pending" | "none";

// The API answers by event id; the evidence path was only ever a proxy for
// it and a freshly pushed alert has none yet.
export function clipRequestKey(event: { id: string | number }): string {
  return String(event.id);
}

// Evidence younger than this with nothing on disk is "still being written".
// Mirrors CLIP_PENDING_WINDOW_S on the API, which is the source of truth
// when it answers; this is the fallback for a reply without the flag.
export const PENDING_WINDOW_S = 120;
export const PENDING_POLL_MS = 3000;

export function evidenceView(
  clip: ClipReply | null | undefined,
  event: { ts: number },
  opts: { videoFailed?: boolean; now?: number } = {},
): EvidenceView {
  const c = clip ?? {};
  const frames = c.frames?.length ? c.frames : [];
  if (c.uri && !opts.videoFailed) return "video";
  if (frames.length) return "frames";
  if (c.uri && opts.videoFailed) return "none";
  const now = opts.now ?? Date.now();
  const age = now / 1000 - (event.ts || 0);
  if (c.pending === true) return "pending";
  if (c.pending === false) return "none";
  return age >= 0 && age < PENDING_WINDOW_S ? "pending" : "none";
}

// A sentence for the card when the video could not be played, or null.
export function playbackProblem(
  clip: ClipReply | null | undefined,
  videoFailed: boolean,
): string | null {
  if (!videoFailed || !clip?.uri) return null;
  const codec = (clip.codec || "").toLowerCase();
  const what =
    codec && codec !== "h264" && codec !== "avc1"
      ? `The clip was saved as ${codec}, which this device cannot play.`
      : "The clip could not be played on this device.";
  return clip.frames?.length
    ? `${what} Showing the recorded frames instead.`
    : `${what} No frames were recorded alongside it.`;
}
