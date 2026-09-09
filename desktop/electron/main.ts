import { app, BrowserWindow, ipcMain, dialog } from "electron";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";
import readline from "node:readline";

const dir = path.dirname(fileURLToPath(import.meta.url));
const root = process.env.ARGUS_REPO || path.resolve(dir, "../..");
let window: BrowserWindow | null = null;
let worker: ChildProcessWithoutNullStreams | undefined;
let sequence = 0;
let exiting = false;
if (process.env.ARGUS_USER_DATA)
  app.setPath("userData", process.env.ARGUS_USER_DATA);
const pending = new Map<
  number,
  {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
    timer: NodeJS.Timeout;
  }
>();
const methods = new Set([
  "auth_state",
  "create_first_owner",
  "sign_in",
  "sign_out",
  "get_site",
  "set_site",
  "list_cameras",
  "add_camera",
  "remove_camera",
  "test",
  "discover_cameras",
  "detect_subnet",
  "scan",
  "list_areas",
  "create_area",
  "assign_camera_area",
  "scene_context",
  "scene_review_summary",
  "area_context",
  "approve_area_context",
  "approve_site_context",
  "update_scene_context",
  "approve_scene_context",
  "request_scene_remap",
  "enqueue_scene_mapping",
  "scene_mapping_progress",
  "accept_suggested_zone",
  "camera_snapshot",
  "list_zones",
  "add_zone",
  "remove_zone",
  "set_camera_rules",
  "presets",
  "use_case_templates",
  "apply_template",
  "add_custom_rule",
  "remove_custom_rule",
  "english_rules_status",
  "list_events",
  "event_clip",
  "acknowledge_alert",
  "resolve_alert",
  "search_events",
  "live_start",
  "camera_links",
  "start_monitoring",
  "stop_monitoring",
  "monitoring_status",
  "feed_sources",
  "switch_feed",
  "feed_switch_status",
  "setup_state",
  "setup_check",
  "mark_configured",
  "send_test_notification",
  "gate_status",
  "pull_model",
  "pull_progress",
  "retention_status",
  "set_retention",
  "list_users",
  "add_user",
  "remove_user",
  "audit_entries",
  "backup_now",
  "download_diagnostics",
  "value_summary",
  "role_table",
  "disk_encryption",
]);

methods.add("live_stop");
function failPending(message: string) {
  for (const item of pending.values()) {
    clearTimeout(item.timer);
    item.reject(new Error(message));
  }
  pending.clear();
}
function ensureWorker() {
  if (worker) return worker;
  const python =
    process.env.ARGUS_PYTHON ||
    path.join(
      root,
      ".venv",
      process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
    );
  if (!fs.existsSync(python))
    throw new Error(
      "Python environment missing. Set ARGUS_PYTHON to the existing Argus Python executable.",
    );
  const bridge = path.resolve(dir, "../bridge.py");
  worker = spawn(
    python,
    [
      "-u",
      bridge,
      "--repo",
      root,
      "--site",
      process.env.ARGUS_SITE_CONFIG || "configs/site_live.json",
      "--db",
      process.env.ARGUS_DB || "runs/desktop/events.db",
    ],
    {
      cwd: root,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
      stdio: "pipe",
    },
  );
  worker.stderr.on("data", (data) => process.stderr.write(data));
  readline.createInterface({ input: worker.stdout }).on("line", (line) => {
    try {
      const data = JSON.parse(line);
      const item = pending.get(data.id);
      if (!item) return;
      clearTimeout(item.timer);
      pending.delete(data.id);
      if (data.error) item.reject(new Error(data.error));
      else item.resolve(data.result);
    } catch {
      console.error("Invalid engine protocol response");
    }
  });
  worker.on("error", (error) => {
    failPending(error.message);
    worker = undefined;
  });
  worker.on("exit", () => {
    failPending("Local backend disconnected. Reconnect to try again.");
    worker = undefined;
  });
  return worker;
}
function invoke(method: string, args: unknown[]) {
  return new Promise((resolve, reject) => {
    try {
      const child = ensureWorker();
      const id = ++sequence;
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(
          new Error(
            "Backend request timed out. Check engine health before retrying a change.",
          ),
        );
      }, 150000);
      pending.set(id, { resolve, reject, timer });
      child.stdin.write(JSON.stringify({ id, method, args }) + "\n");
    } catch (error) {
      reject(error);
    }
  });
}

app.whenReady().then(() => {
  ipcMain.handle("engine:environment", (event) => {
    if (event.sender !== window?.webContents) throw new Error("Unknown caller");
    return {
      repo: root,
      site: process.env.ARGUS_SITE_CONFIG || "configs/site_live.json",
      db: process.env.ARGUS_DB || "runs/desktop/events.db",
    };
  });
  ipcMain.handle("engine:invoke", (event, method, args) => {
    if (
      event.sender !== window?.webContents ||
      event.senderFrame !== window.webContents.mainFrame
    )
      throw new Error("Unknown caller");
    if (
      typeof method !== "string" ||
      !methods.has(method) ||
      !Array.isArray(args) ||
      JSON.stringify(args).length > 1000000
    )
      throw new Error("Invalid engine request");
    return invoke(method, args);
  });
  const create = () => {
    window = new BrowserWindow({
      width: 1480,
      height: 960,
      minWidth: 860,
      minHeight: 640,
      backgroundColor: "#f5f6f4",
      title: "ARGUS",
      webPreferences: {
        preload: path.join(dir, "preload.cjs"),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
      },
    });
    window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    window.webContents.on("will-navigate", (event) => event.preventDefault());
    if (process.argv.includes("--production"))
      void window.loadFile(path.join(dir, "../dist/index.html"));
    else void window.loadURL("http://127.0.0.1:5173");
  };
  create();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) create();
  });
});
app.on("before-quit", (event) => {
  if (exiting || !worker) return;
  event.preventDefault();
  exiting = true;
  const timeout = setTimeout(() => {
    worker?.kill();
    app.quit();
  }, 10000);
  void invoke("shutdown", []).finally(() => {
    clearTimeout(timeout);
    worker?.kill();
    app.quit();
  });
});
app.on("window-all-closed", () => app.quit());
process.on("uncaughtException", (error) => {
  dialog.showErrorBox("Argus desktop", error.message);
});
