import { useEffect, useState } from "react";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Plus,
  RefreshCw,
} from "lucide-react";
import type { Auth, Camera, Hierarchy, Json, Mode, Transport } from "../lib/types";
import { Badge, Notice, Spinner } from "./common";
import VerifierDownload from "./VerifierDownload";
import LocationManager from "./LocationManager";
import NotificationSetup from "./NotificationSetup";
// Locations come FIRST: Add camera asks which branch and area a camera belongs
// to, and on a new site that list is empty, so the wizard used to demand a
// choice from nothing ("the flow u added for setup is contradicting" -- Martin,
// 20 Sep). Alerts get their own step because a site that detects everything and
// tells nobody is not set up.
const steps = [
  "Locations",
  "Cameras",
  "Scenes & zones",
  "Use case",
  "Detectors",
  "Alerts",
  "Verification",
  "Finish",
];
export default function Setup({
  api,
  mode,
  cameras,
  canConfigureCameras,
  auth,
  hierarchy,
  site,
  notify,
  onAdd,
  onConfigure,
  onChange,
  onFinish,
}: {
  api: Transport;
  mode: Mode;
  cameras: Camera[];
  canConfigureCameras: boolean;
  auth: Auth;
  hierarchy: Hierarchy;
  site: Json;
  notify: (message: string) => void;
  onAdd?: () => void;
  onConfigure: (c: Camera, tab?: string) => void;
  onChange: () => Promise<void>;
  onFinish: () => void;
}) {
  const [step, setStep] = useState(0);
  const [templates, setTemplates] = useState<Json>({});
  const [template, setTemplate] = useState("");
  const [checks, setChecks] = useState<Json[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState("");
  useEffect(() => {
    api
      .invoke("use_case_templates")
      .then(setTemplates)
      .catch((e) => setError(e.message));
  }, [api]);
  async function run(method: string, args: unknown[] = []) {
    setBusy(true);
    setError("");
    try {
      const r = await api.invoke(method, args);
      if (r.ok === false) throw new Error(r.message || "Operation failed");
      if (method === "setup_check") setChecks(r);
      else if (method === "mark_configured") onFinish();
      else
        setResult(
          method === "send_test_notification"
            ? `Test sent via ${r.via}. Confirm receipt at the destination.`
            : "Settings saved.",
        );
      await onChange();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="setup-layout">
      <ol className="setup-steps">
        {steps.map((s, i) => (
          <li
            key={s}
            className={i === step ? "active" : i < step ? "complete" : ""}
          >
            <button onClick={() => setStep(i)}>
              <span>{i < step ? <Check size={14} /> : i + 1}</span>
              {s}
            </button>
          </li>
        ))}
      </ol>
      <section className="setup-content">
        {/* The 3.3 GB model is the slowest part of setup, so it starts here at
            step 0 and downloads WHILE the operator works, not after. */}
        <VerifierDownload api={api} mode={mode} autoStart />
        {error && <Notice error>{error}</Notice>}
        {result && <Notice>{result}</Notice>}
        <span className="eyebrow">STEP {step + 1} OF {steps.length}</span>
        <h2>{steps[step]}</h2>
        {step === 0 && (
          <>
            <p>
              Name the branches and areas of your site first, so a camera has
              somewhere to go when you add it. You can skip this and place
              cameras later.
            </p>
            <LocationManager
              api={api}
              mode={mode}
              auth={auth}
              hierarchy={hierarchy}
              cameras={cameras}
              onChange={onChange}
              notify={notify}
            />
          </>
        )}
        {step === 1 && (
          <>
            <p>Connect cameras and assign them to the areas of your site.</p>
            {canConfigureCameras && onAdd && (
              <button className="button primary" onClick={onAdd}>
                <Plus size={16} />
                Add camera
              </button>
            )}
          </>
        )}
        {[1, 2, 4].includes(step) && (
          <div className="setup-cameras">
            {cameras.map((c) => (
              <div className="setting-row" key={c.id}>
                <div>
                  <strong>{c.id}</strong>
                  <small>{c.area_id || "Ungrouped"}</small>
                </div>
                <button
                  className="button"
                  onClick={() =>
                    onConfigure(c, step === 4 ? "detectors" : "scene")
                  }
                >
                  {step === 4 ? "Configure detectors" : "Review scene"}
                </button>
                {step === 2 && (
                  <button
                    className="button"
                    onClick={() => onConfigure(c, "zones")}
                  >
                    Draw zone
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
        {step === 2 && (
          <Notice>
            Each camera needs its own evidence. Review the scene and correct it
            before approval, even when cameras share an area.
          </Notice>
        )}
        {step === 3 && (
          <>
            <label>
              Site use case
              <select
                value={template}
                onChange={(e) => setTemplate(e.target.value)}
              >
                <option value="">Choose a use case</option>
                {Object.entries(templates).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v.label || v.name || k}
                  </option>
                ))}
              </select>
            </label>
            <p className="muted">
              Applying a template changes detector settings for the site.
              Existing zone and English rules remain managed by the backend.
            </p>
            <button
              className="button primary"
              disabled={!template || busy}
              onClick={() => void run("apply_template", [template])}
            >
              Apply use case
            </button>
          </>
        )}
        {step === 5 && (
          <>
            <p>
              Choose where an alert goes. A site that detects everything and
              tells nobody is not set up.
            </p>
            <NotificationSetup
              api={api}
              mode={mode}
              site={site}
              onChange={onChange}
              notify={notify}
            />
          </>
        )}
        {step === 6 && (
          <>
            <p>
              Check camera connectivity, detector weights and the local vision
              verifier.
            </p>
            {mode === "demo" ? (
              <Notice>
                Live readiness checks are unavailable in demo mode. This
                walkthrough does not validate an AI installation.
              </Notice>
            ) : (
              <button
                className="button"
                disabled={busy}
                onClick={() => void run("setup_check")}
              >
                {busy ? <Spinner /> : <RefreshCw size={16} />}Run readiness
                checks
              </button>
            )}
            <VerifierDownload api={api} mode={mode} />
            {checks.map((c) => (
              <div className="setting-row" key={c.id}>
                <div>
                  <strong>{c.label}</strong>
                  <small>{c.detail}</small>
                  {c.fix && <small>{c.fix}</small>}
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
          </>
        )}
        {step === 7 && mode === "engine" && (
          <>
            {!checks.length && (
              <Notice>
                Readiness checks have not been run. You can finish now and run
                them later from Settings; anything that needs attention will be
                shown there.
              </Notice>
            )}
            {checks.some((c) => c.ok === false) && (
              <Notice error>
                You can finish now. These still need attention:{" "}
                {checks
                  .filter((c) => c.ok === false)
                  .map((c) => c.label)
                  .join(", ")}
                .
              </Notice>
            )}
          </>
        )}
        {step === 7 && (
          <>
            <h3>
              {mode === "demo"
                ? "Demo walkthrough complete."
                : "Review before going live."}
            </h3>
            <p>
              {cameras.length} cameras configured. Monitoring starts only when
              you choose Start monitoring in the overview.
            </p>
            {mode === "engine" && (
              <>
                <button
                  className="button"
                  disabled={busy}
                  onClick={() => void run("send_test_notification")}
                >
                  Send test notification
                </button>
                {checks.some((c) => c.ok === false) && (
                  <Notice error>
                    Some readiness checks need attention. Resolve them before
                    relying on monitoring.
                  </Notice>
                )}
              </>
            )}
          </>
        )}
        <div className="setup-footer">
          <button
            className="button"
            disabled={step === 0 || busy}
            onClick={() => {
              setResult("");
              setStep(step - 1);
            }}
          >
            <ChevronLeft size={16} />
            Back
          </button>
          {step < 5 ? (
            <button
              className="button primary"
              disabled={busy || (!cameras.length && step === 1)}
              onClick={() => {
                setResult("");
                setStep(step + 1);
              }}
            >
              Continue
              <ChevronRight size={16} />
            </button>
          ) : (
            <button
              className="button primary"
              // Readiness checks INFORM; they do not lock the door. This used
              // to be disabled until every check passed, so an installed site
              // whose AI model had not finished its 3.3 GB download had a
              // Finish button that did nothing, with no word as to why
              // (Martin, 21 Sep: "finish setup button wasn't clicking"). The
              // engine runs generic and loud without the model by design --
              // complete_first_run says so in its own docstring.
              disabled={busy || !cameras.length}
              title={
                !cameras.length ? "Add at least one camera first" : undefined
              }
              onClick={() => void run("mark_configured")}
            >
              Finish setup
              <Check size={16} />
            </button>
          )}
        </div>
      </section>
    </div>
  );
}
