import path from "node:path";
import fs from "node:fs";
import os from "node:os";

/**
 * Where everything lives, in development and in an installed app.
 *
 * Development runs out of the repo: a .venv interpreter, `python -m cvti.api`,
 * paths relative to the checkout. An installed customer has none of that — no
 * repo, no venv, no Python — so the shell spawns the frozen `argus-api`
 * binary that PyInstaller puts beside the engine, and the writable files live
 * in the per-user data directory rather than inside a read-only app bundle.
 *
 * The data directory deliberately matches `cvti/utils.py: user_data_dir()`
 * exactly. Both halves of the product must agree on it, and an upgrade from
 * the old PyQt shell has to find the site config, accounts and events it
 * already had.
 */
export type InstallLayout = {
  packaged: boolean;
  /** Working directory for the spawned backend; holds configs/, models/, vendor/. */
  engineRoot: string;
  /** Executable to spawn for the Engine API. */
  apiCommand: string;
  /** Args that precede the API's own flags ([] for the frozen binary). */
  apiArgs: string[];
  /** Writable per-user directory, shared with the Python side. */
  dataDir: string;
  site: string;
  db: string;
  supportLog: string;
};

export function userDataDir(
  platform: string = process.platform,
  env: NodeJS.ProcessEnv = process.env,
  home: string = os.homedir(),
): string {
  if (platform === "darwin")
    return path.join(home, "Library", "Application Support", "Argus");
  if (platform === "win32") return path.join(env.APPDATA || home, "Argus");
  return path.join(home, ".argus");
}

export type LayoutInput = {
  packaged: boolean;
  /** Directory of the compiled main process (dist-electron). */
  dir: string;
  /** Electron's resourcesPath; only read when packaged. */
  resourcesPath: string;
  env?: NodeJS.ProcessEnv;
  platform?: string;
  home?: string;
  exists?: (candidate: string) => boolean;
};

export function resolveLayout({
  packaged,
  dir,
  resourcesPath,
  env = process.env,
  platform = process.platform,
  home = os.homedir(),
  exists = fs.existsSync,
}: LayoutInput): InstallLayout {
  const exe = platform === "win32" ? ".exe" : "";
  // Development: the repo root two levels above dist-electron.
  const repoRoot = env.ARGUS_REPO || path.resolve(dir, "../..");
  // Installed: electron-builder copies PyInstaller's output tree to
  // resources/engine, so the API binary, the engine binary, configs/ and
  // models/ all sit together there.
  const engineRoot = packaged ? path.join(resourcesPath, "engine") : repoRoot;

  let apiCommand: string;
  let apiArgs: string[];
  if (packaged) {
    apiCommand = env.ARGUS_API_BIN || path.join(engineRoot, `argus-api${exe}`);
    apiArgs = [];
  } else {
    apiCommand =
      env.ARGUS_PYTHON ||
      path.join(
        repoRoot,
        ".venv",
        platform === "win32" ? "Scripts/python.exe" : "bin/python",
      );
    apiArgs = ["-u", "-m", "cvti.api"];
  }
  if (!exists(apiCommand))
    throw new Error(
      packaged
        ? `The Argus engine is missing from this installation (expected ${apiCommand}). Reinstall Argus.`
        : "Python environment missing. Set ARGUS_PYTHON to the existing Argus Python executable.",
    );

  // Writable state. In development it stays in the checkout, where the repo's
  // own tooling expects it. Installed, it goes to <user data>/site — the exact
  // paths cvti/app/shell.py already uses when frozen, so upgrading from the
  // PyQt shell to this one keeps the site config, accounts and events that
  // machine already had instead of silently starting empty.
  const dataDir = packaged
    ? path.join(userDataDir(platform, env, home), "site")
    : repoRoot;
  const rel = (value: string) =>
    path.isAbsolute(value) ? value : path.join(dataDir, value);
  return {
    packaged,
    engineRoot,
    apiCommand,
    apiArgs,
    dataDir,
    site: rel(env.ARGUS_SITE_CONFIG || (packaged ? "site.json" : "configs/site_live.json")),
    db: rel(env.ARGUS_DB || (packaged ? "events.db" : "runs/desktop/events.db")),
    supportLog: rel(
      env.ARGUS_SUPPORT_LOG ||
        (packaged ? "frontend.log" : "runs/desktop/frontend.log"),
    ),
  };
}
