import { describe, expect, it } from "vitest";
import path from "node:path";
import { resolveLayout, userDataDir } from "../electron/install-layout.js";

/**
 * The installed app and the development checkout are different machines'
 * worth of assumptions. Until v1.8.13 only the development one existed, and
 * every release shipped the old PyQt interface because nothing packaged this
 * shell at all. These are the rules that make an installed build work.
 */
const base = {
  dir: "/opt/Argus/resources/app.asar/dist-electron",
  resourcesPath: "/opt/Argus/resources",
};

describe("installed", () => {
  const packaged = (env: NodeJS.ProcessEnv = {}, platform = "linux") =>
    resolveLayout({
      ...base,
      packaged: true,
      platform,
      home: "/home/x",
      env,
      exists: () => true,
    });

  it("spawns the frozen binary beside the engine, with no module args", () => {
    const layout = packaged();
    expect(layout.apiCommand).toBe("/opt/Argus/resources/engine/argus-api");
    // The frozen binary IS cvti.api; passing `-m cvti.api` to it would make it
    // parse "cvti.api" as a flag and die.
    expect(layout.apiArgs).toEqual([]);
    expect(layout.engineRoot).toBe("/opt/Argus/resources/engine");
  });

  it("names the Windows executable and reads APPDATA", () => {
    const layout = packaged({ APPDATA: "C:\\Users\\m\\AppData\\Roaming" }, "win32");
    expect(layout.apiCommand).toContain("argus-api.exe");
    expect(layout.dataDir).toBe(
      path.join("C:\\Users\\m\\AppData\\Roaming", "Argus", "site"),
    );
  });

  it("writes where the PyQt shell already wrote, so an upgrade keeps its data", () => {
    // cvti/app/shell.py (frozen): user_data_dir()/site/{site.json,events.db}.
    // Disagreeing here would silently start every upgraded install empty.
    const layout = packaged({}, "darwin");
    const site = path.join(
      "/home/x",
      "Library",
      "Application Support",
      "Argus",
      "site",
    );
    expect(layout.dataDir).toBe(site);
    expect(layout.site).toBe(path.join(site, "site.json"));
    expect(layout.db).toBe(path.join(site, "events.db"));
  });

  it("never writes inside the read-only app bundle", () => {
    const layout = packaged();
    for (const p of [layout.site, layout.db, layout.supportLog])
      expect(p.startsWith(layout.engineRoot)).toBe(false);
  });

  it("says the app is damaged when its engine is missing, not that Python is", () => {
    expect(() =>
      resolveLayout({ ...base, packaged: true, env: {}, exists: () => false }),
    ).toThrow(/Reinstall Argus/);
  });

  it("honours explicit overrides", () => {
    const layout = packaged({
      ARGUS_API_BIN: "/tmp/fake-api",
      ARGUS_DB: "/tmp/events.db",
    });
    expect(layout.apiCommand).toBe("/tmp/fake-api");
    expect(layout.db).toBe("/tmp/events.db");
  });
});

describe("development", () => {
  const dev = (env: NodeJS.ProcessEnv = {}) =>
    resolveLayout({
      dir: "/repo/Frontend/dist-electron",
      resourcesPath: "/unused",
      packaged: false,
      platform: "linux",
      home: "/home/x",
      env,
      exists: () => true,
    });

  it("runs the repo's interpreter as a module, unchanged", () => {
    const layout = dev();
    expect(layout.apiCommand).toBe("/repo/.venv/bin/python");
    expect(layout.apiArgs).toEqual(["-u", "-m", "cvti.api"]);
    expect(layout.engineRoot).toBe("/repo");
    expect(layout.site).toBe("/repo/configs/site_live.json");
    expect(layout.db).toBe("/repo/runs/desktop/events.db");
  });

  it("still asks for ARGUS_PYTHON when there is no venv", () => {
    expect(() =>
      resolveLayout({
        dir: "/repo/Frontend/dist-electron",
        resourcesPath: "/unused",
        packaged: false,
        env: {},
        exists: () => false,
      }),
    ).toThrow(/ARGUS_PYTHON/);
  });
});

describe("userDataDir", () => {
  it("matches cvti/utils.py on every platform", () => {
    expect(userDataDir("darwin", {}, "/h")).toBe(
      "/h/Library/Application Support/Argus",
    );
    expect(userDataDir("win32", { APPDATA: "C:\\a" }, "/h")).toBe(
      path.join("C:\\a", "Argus"),
    );
    expect(userDataDir("linux", {}, "/h")).toBe("/h/.argus");
  });
});
