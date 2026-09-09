import { useEffect, useState } from "react";
import { Save, RefreshCw, ShieldCheck } from "lucide-react";
import type { Json, Mode, Transport } from "../lib/types";
import { Badge, Notice, Spinner } from "./common";
export default function SettingsPanel({
  api,
  mode,
  site,
  onChange,
  notify,
}: {
  api: Transport;
  mode: Mode;
  site: Json;
  onChange: () => Promise<void>;
  notify: (s: string) => void;
}) {
  const [name, setName] = useState(site.name || "");
  const [notification, setNotification] = useState(site.notify || "console");
  const [days, setDays] = useState(site.retention_days || 30);
  const [gate, setGate] = useState<Json>({});
  const [checks, setChecks] = useState<Json[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [env, setEnv] = useState<Json>({});
  useEffect(() => {
    api
      .invoke("gate_status")
      .then(setGate)
      .catch((e) => setError(e.message));
    if (mode === "engine")
      window.argusDesktop
        ?.environment()
        .then(setEnv)
        .catch((e) => setError(e.message));
  }, [api, mode]);
  async function run(method: string, args: unknown[] = [], message?: string) {
    setBusy(true);
    setError("");
    try {
      const r = await api.invoke(method, args);
      if (r.ok === false) throw new Error(r.message || "Operation failed");
      if (method === "gate_status") setGate(r);
      if (method === "setup_check") setChecks(r);
      await onChange();
      if (message) notify(message);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="settings-layout">
      {error && <Notice error>{error}</Notice>}
      <section className="settings-section">
        <div>
          <h2>Site identity</h2>
          <p>The installation and its notification destination.</p>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void run("set_site", [name, notification], "Site settings saved");
          }}
        >
          <label>
            Site name
            <input
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label>
            Notification destination
            <input
              value={notification}
              onChange={(e) => setNotification(e.target.value)}
              placeholder="console"
              autoComplete="off"
            />
          </label>
          <button className="button primary" disabled={busy}>
            <Save size={16} />
            Save site
          </button>
          {mode === "engine" && (
            <button
              className="button"
              type="button"
              disabled={busy}
              onClick={() =>
                void run(
                  "send_test_notification",
                  [],
                  "Test sent. Check the configured destination for receipt.",
                )
              }
            >
              Send test
            </button>
          )}
        </form>
      </section>
      <section className="settings-section">
        <div>
          <h2>AI verification</h2>
          <p>Local vision model and runtime availability.</p>
        </div>
        <div>
          <div className="gate-status">
            <ShieldCheck size={24} />
            <div>
              <strong>
                {mode === "demo"
                  ? "No verifier connected"
                  : gate.model || "Local verifier"}
              </strong>
              <small>
                {mode === "demo"
                  ? "Demo mode does not perform inference"
                  : gate.mode || "Status unavailable"}
              </small>
            </div>
            <Badge tone={gate.mode === "live" ? "green" : "amber"}>
              {gate.mode === "live" ? "Available" : "Not verified"}
            </Badge>
          </div>
          {mode === "engine" && (
            <div className="actions">
              <button
                className="button"
                disabled={busy}
                onClick={() => void run("gate_status")}
              >
                <RefreshCw size={16} />
                Recheck verifier
              </button>
              <button
                className="button"
                disabled={busy}
                onClick={() => void run("setup_check")}
              >
                {busy ? <Spinner /> : <CheckIcon />}Run system checks
              </button>
            </div>
          )}
          {checks.map((c) => (
            <div className="setting-row" key={c.id}>
              <div>
                <strong>{c.label}</strong>
                <small>{c.detail}</small>
                <small>{c.fix}</small>
              </div>
              <Badge
                tone={
                  c.ok === true ? "green" : c.ok === false ? "red" : "amber"
                }
              >
                {c.ok === true
                  ? "Ready"
                  : c.ok === false
                    ? "Action needed"
                    : "Warning"}
              </Badge>
            </div>
          ))}
        </div>
      </section>
      <section className="settings-section">
        <div>
          <h2>Evidence retention</h2>
          <p>Retention policy used by the existing engine.</p>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void run("set_retention", [days], "Retention policy saved");
          }}
        >
          <label>
            Retention days
            <input
              type="number"
              min="1"
              max="3650"
              required
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              disabled={mode === "demo"}
            />
          </label>
          <button className="button" disabled={busy || mode === "demo"}>
            <Save size={16} />
            Save policy
          </button>
          {mode === "demo" && (
            <p className="field-note">
              Unavailable in demo mode; no evidence files are managed here.
            </p>
          )}
        </form>
      </section>
      {mode === "engine" && (
        <section className="settings-section">
          <div>
            <h2>Installation</h2>
            <p>Source-run backend connection.</p>
          </div>
          <dl className="facts">
            <dt>Repository</dt>
            <dd>{env.repo || "Unavailable"}</dd>
            <dt>Site configuration</dt>
            <dd>{env.site || "Unavailable"}</dd>
            <dt>Events database</dt>
            <dd>{env.db || "Unavailable"}</dd>
          </dl>
        </section>
      )}
    </div>
  );
}
function CheckIcon() {
  return <ShieldCheck size={16} />;
}
