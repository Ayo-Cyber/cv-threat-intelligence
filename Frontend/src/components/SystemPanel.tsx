import { useState } from "react";
import { Play, RefreshCw } from "lucide-react";
import type { Json, Transport } from "../lib/types";
import { Notice, Spinner } from "./common";

/**
 * Everything the retired PyQt console could do that this UI could not.
 *
 * The React shell shipped with screens for the daily work — cameras, zones,
 * incidents, English rules — and no way at all to reach 42 other backend
 * capabilities: model downloads, backups, audit export, account recovery,
 * reports, heartbeat, learning stats. The methods existed, the API routes did
 * not, and nothing called them. An operator could not test or even see them.
 *
 * This is deliberately a plain operations console, not 29 bespoke screens:
 * one row per capability, real arguments, the raw reply shown as returned. It
 * makes every capability reachable and testable now; a capability that earns
 * a designed screen can graduate to one later.
 */

type Field = { name: string; label: string; placeholder?: string; kind?: "number" | "bool" | "json" };
type Action = {
  method: string;
  label: string;
  hint?: string;
  fields?: Field[];
  danger?: boolean;
};
type Group = { title: string; blurb: string; actions: Action[] };

const GROUPS: Group[] = [
  {
    title: "Engine & models",
    blurb:
      "The verification model downloads on first use. Without this, alerts arrive unverified and nothing in the UI could start the download.",
    actions: [
      { method: "app_version", label: "Build version" },
      { method: "detector_validation", label: "Validate detectors" },
      { method: "live_frames", label: "Live frame publishers" },
      {
        method: "pull_model",
        label: "Download verification model",
        hint: "Several GB on first run; needs internet.",
      },
      { method: "pull_progress", label: "Download progress" },
      { method: "learning_stats", label: "Learning stats" },
      { method: "learning_calibrate", label: "Recalibrate thresholds" },
    ],
  },
  {
    title: "Reports",
    blurb: "Summaries the old console produced for handover and review.",
    actions: [
      { method: "counts", label: "Event counts" },
      {
        method: "needs_attention",
        label: "Needs attention",
        fields: [{ name: "min_priority", label: "Min priority", placeholder: "medium" }],
      },
      { method: "weekly_summary", label: "Weekly summary" },
      {
        method: "handover",
        label: "Shift handover",
        fields: [{ name: "hours", label: "Hours", placeholder: "8", kind: "number" }],
      },
    ],
  },
  {
    title: "Evidence",
    blurb: "Export and retention holds for a specific incident.",
    actions: [
      {
        method: "export_incident_pdf",
        label: "Incident PDF",
        fields: [{ name: "event_id", label: "Event id", placeholder: "123" }],
      },
      {
        method: "export_evidence",
        label: "Export evidence",
        fields: [
          { name: "event_ids", label: "Event ids", placeholder: "comma separated, blank = all" },
          { name: "dest", label: "Destination", placeholder: "blank = default folder" },
        ],
      },
      {
        method: "set_legal_hold",
        label: "Legal hold",
        hint: "Exempts an incident from retention deletion.",
        fields: [
          { name: "event_id", label: "Event id", placeholder: "123", kind: "number" },
          { name: "hold", label: "Hold", placeholder: "true", kind: "bool" },
        ],
      },
    ],
  },
  {
    title: "Backups",
    blurb: "Configuration and evidence backups.",
    actions: [
      { method: "list_backups", label: "List backups" },
      { method: "backup_now", label: "Back up now" },
      {
        method: "set_backup_dir",
        label: "Set backup folder",
        fields: [{ name: "path", label: "Folder", placeholder: "/path/to/backups" }],
      },
      {
        method: "restore_backup",
        label: "Restore from backup",
        danger: true,
        hint: "Replaces current configuration.",
        fields: [{ name: "zip_path", label: "Backup zip", placeholder: "/path/to/backup.zip" }],
      },
    ],
  },
  {
    title: "Audit & compliance",
    blurb: "The tamper-evident log, and where evidence is stored.",
    actions: [
      { method: "audit_verify", label: "Verify audit chain" },
      { method: "audit_export", label: "Export audit log" },
      { method: "disk_encryption", label: "Disk encryption status" },
      { method: "download_diagnostics", label: "Download diagnostics" },
    ],
  },
  {
    title: "Uptime monitoring",
    blurb: "Pings an external monitor so a dead engine is noticed off-box.",
    actions: [
      { method: "heartbeat_status", label: "Heartbeat status" },
      {
        method: "set_heartbeat",
        label: "Configure heartbeat",
        fields: [
          { name: "url", label: "URL", placeholder: "https://… (blank disables)" },
          { name: "key", label: "Key", placeholder: "optional" },
        ],
      },
    ],
  },
  {
    title: "Accounts",
    blurb:
      "Sign-in and adding users already worked. The rest of the lifecycle had no screen, so a locked-out site had no way back in.",
    actions: [
      { method: "auth_accounts", label: "List accounts" },
      { method: "role_table", label: "Roles & permissions" },
      { method: "auth_recovery", label: "Recovery instructions" },
      {
        method: "change_own_password",
        label: "Change my password",
        fields: [
          { name: "current", label: "Current", placeholder: "current password" },
          { name: "new", label: "New", placeholder: "12+ characters" },
        ],
      },
      {
        method: "set_user_role",
        label: "Set a user's role",
        fields: [
          { name: "username", label: "User", placeholder: "username" },
          { name: "role", label: "Role", placeholder: "operator | installer | owner" },
        ],
      },
      {
        method: "remove_user",
        label: "Remove a user",
        danger: true,
        fields: [{ name: "username", label: "User", placeholder: "username" }],
      },
      {
        method: "create_owner_override",
        label: "Create owner (override)",
        danger: true,
        hint: "Last resort when no owner can sign in.",
        fields: [
          { name: "username", label: "User", placeholder: "username" },
          { name: "password", label: "Password", placeholder: "12+ characters" },
        ],
      },
    ],
  },
  {
    title: "Network discovery",
    blurb: "Find cameras on this site's network.",
    actions: [
      { method: "detect_subnet", label: "Detect subnet" },
      { method: "scan", label: "Scan for cameras" },
    ],
  },
  {
    title: "Per-camera rules",
    blurb:
      "The English-rule box lives inside a camera. These are the same controls, reachable without hunting for the camera first.",
    actions: [
      {
        method: "set_custom_rule",
        label: "Set the camera's question",
        fields: [
          { name: "camera_id", label: "Camera", placeholder: "camera id" },
          { name: "question", label: "Question", placeholder: "a person without a hard hat" },
          { name: "dwell", label: "Dwell (s)", placeholder: "4", kind: "number" },
        ],
      },
      {
        method: "add_custom_threat",
        label: "Add a named threat",
        fields: [
          { name: "camera_id", label: "Camera", placeholder: "camera id" },
          { name: "name", label: "Name", placeholder: "no_hard_hat" },
          { name: "description", label: "Description", placeholder: "a person without a hard hat" },
        ],
      },
      {
        method: "remove_custom_threat",
        label: "Remove a named threat",
        fields: [
          { name: "camera_id", label: "Camera", placeholder: "camera id" },
          { name: "index", label: "Index", placeholder: "0", kind: "number" },
        ],
      },
      {
        method: "camera_links",
        label: "Camera links",
        fields: [{ name: "camera_id", label: "Camera", placeholder: "camera id" }],
      },
    ],
  },
  {
    title: "Site & scene",
    blurb: "Scene context drives which rules a camera is allowed to act on.",
    actions: [
      {
        method: "update_site_context",
        label: "Update site context",
        hint: "JSON object; the backend validates required fields.",
        fields: [{ name: "context", label: "Context JSON", placeholder: '{"confidence": "high"}', kind: "json" }],
      },
      {
        method: "area_context",
        label: "Area context",
        fields: [{ name: "area_id", label: "Area", placeholder: "area id" }],
      },
      { method: "scene_mapping_progress", label: "Scene mapping progress" },
      {
        method: "live_start",
        label: "Start a live view",
        hint: "Per-camera streams normally start from the wall; this is the raw call.",
        fields: [{ name: "camera_id", label: "Camera", placeholder: "camera id" }],
      },
    ],
  },
  {
    title: "Value reporting",
    blurb: "The figures behind the value summary.",
    actions: [
      {
        method: "set_value_inputs",
        label: "Set value inputs",
        fields: [
          { name: "incident_value", label: "Incident value", placeholder: "500", kind: "number" },
          { name: "guard_hourly_cost", label: "Guard cost/hr", placeholder: "12", kind: "number" },
          { name: "review_minutes", label: "Review minutes", placeholder: "5", kind: "number" },
        ],
      },
    ],
  },
];

function coerce(field: Field, raw: string): unknown {
  if (raw === "") return undefined;
  if (field.kind === "number") {
    const n = Number(raw);
    if (Number.isNaN(n)) throw new Error(`${field.label} must be a number`);
    return n;
  }
  if (field.kind === "bool") return !/^(false|no|0)$/i.test(raw.trim());
  if (field.kind === "json") {
    try {
      return JSON.parse(raw);
    } catch {
      throw new Error(`${field.label} must be valid JSON`);
    }
  }
  return raw;
}

function ActionRow({
  action,
  api,
  notify,
}: {
  action: Action;
  api: Transport;
  notify: (s: string) => void;
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<Json | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const args = (action.fields ?? []).map((f) => coerce(f, values[f.name] ?? ""));
      // Trailing omitted optionals stay off the call so the backend's own
      // defaults apply, rather than being overridden with undefined.
      while (args.length && args[args.length - 1] === undefined) args.pop();
      const out = await api.invoke<Json>(action.method, args);
      setResult(out ?? { ok: true });
      notify(`${action.label}: done`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="system-action">
      <div className="system-action-head">
        <div>
          <strong>{action.label}</strong>
          <code>{action.method}</code>
          {action.hint && <small>{action.hint}</small>}
        </div>
        <button
          type="button"
          className={action.danger ? "danger" : ""}
          onClick={run}
          disabled={busy}
        >
          {busy ? <Spinner /> : <Play size={14} />}
          Run
        </button>
      </div>
      {action.fields && action.fields.length > 0 && (
        <div className="system-action-fields">
          {action.fields.map((f) => (
            <label key={f.name}>
              {f.label}
              <input
                value={values[f.name] ?? ""}
                placeholder={f.placeholder}
                onChange={(e) =>
                  setValues((v) => ({ ...v, [f.name]: e.target.value }))
                }
              />
            </label>
          ))}
        </div>
      )}
      {error && <Notice error>{error}</Notice>}
      {result !== null && (
        <pre className="system-result">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}

export default function SystemPanel({
  api,
  notify,
}: {
  api: Transport;
  notify: (s: string) => void;
}) {
  const [open, setOpen] = useState<string>(GROUPS[0].title);
  return (
    <section className="card system-panel">
      <header>
        <h2>
          <RefreshCw size={18} /> System operations
        </h2>
        <p>
          Backend capabilities that have no dedicated screen yet. Arguments are
          passed through and the reply is shown exactly as the engine returned
          it.
        </p>
      </header>
      {GROUPS.map((g) => (
        <div key={g.title} className="system-group">
          <button
            type="button"
            className="system-group-head"
            onClick={() => setOpen(open === g.title ? "" : g.title)}
            aria-expanded={open === g.title}
          >
            <strong>{g.title}</strong>
            <span>{g.actions.length}</span>
          </button>
          {open === g.title && (
            <div className="system-group-body">
              <p className="system-blurb">{g.blurb}</p>
              {g.actions.map((a) => (
                <ActionRow key={a.method} action={a} api={api} notify={notify} />
              ))}
            </div>
          )}
        </div>
      ))}
    </section>
  );
}
