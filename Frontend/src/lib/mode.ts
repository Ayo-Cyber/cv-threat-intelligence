import type { Mode } from "./types";

const KEY = "argus.workspace.mode";

/** The desktop app IS the product: it opens on the real workspace, which means
 * the sign-in / create-owner screen on a fresh install. Demo is a sandbox for
 * the browser preview, where there is no engine to talk to. Before this, every
 * launch landed in demo — a hardcoded fixture site — so a first-time user never
 * saw sign-up or site setup at all. */
export function initialMode(
  storage: Pick<Storage, "getItem"> | null,
  desktop: boolean,
): Mode {
  if (!desktop) return "demo";
  let stored: string | null = null;
  try {
    stored = storage?.getItem(KEY) ?? null;
  } catch {
    stored = null;
  }
  return stored === "demo" || stored === "engine" ? stored : "engine";
}

export function rememberMode(
  storage: Pick<Storage, "setItem"> | null,
  mode: Mode,
): void {
  try {
    storage?.setItem(KEY, mode);
  } catch {
    /* private browsing, quota, or no storage at all — the default still holds */
  }
}
