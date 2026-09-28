import { useCallback, useEffect, useState } from "react";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Plus,
  RefreshCw,
} from "lucide-react";
import type { Auth, Camera, Hierarchy, Json, Mode, Transport } from "../lib/types";
import { Badge, Notice, Spinner } from "./common";
import VerifierDownload, {
  MODEL_SIZE,
  type GateStatus,
  type PullProgress,
} from "./VerifierDownload";
import LocationManager from "./LocationManager";
import NotificationSetup from "./NotificationSetup";
import {
  SETUP_STEPS,
  finishBlockedReason,
  modelReady,
  stepIndex,
} from "../lib/setup-flow";

// The order is the point (lib/setup-flow.ts says why): the AI model starts
// downloading FIRST and keeps going while locations and cameras are set up;
// Finish waits for it, and says so, instead of a Skip that left sites with
// cameras nobody mapped ("agent mapping wont run", 28 Sep). Locations still
// come before cameras: Add camera asks which area a camera belongs to, and
// on a new site that list is empty (Martin, 20 Sep).
const STEP_AI = stepIndex("AI model");
const STEP_LOCATIONS = stepIndex("Locations");
const STEP_CAMERAS = stepIndex("Cameras");
const STEP_DETECTORS = stepIndex("Detectors");
const STEP_ALERTS = stepIndex("Alerts");
const STEP_FINISH = stepIndex("Finish");

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
  // The model's state, reported by whichever VerifierDownload is mounted.
  const [gate, setGate] = useState<GateStatus | null>(null);
  const [pull, setPull] = useState<PullProgress | null>(null);
  const [online, setOnline] = useState(
    typeof navigator === "undefined" ? true : navigator.onLine !== false,
  );
  const onStatus = useCallback((g: GateStatus | null, p: PullProgress | null) => {
    setGate(g);
    setPull(p);
  }, []);
  useEffect(() => {
    const up = () => setOnline(true);
    const down = () => setOnline(false);
    window.addEventListener("online", up);
    window.addEventListener("offline", down);
    return () => {
      window.removeEventListener("online", up);
      window.removeEventListener("offline", down);
    };
  }, []);
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
  const ready = mode === "demo" || modelReady(gate);
  const blocked = finishBlockedReason({
    mode,
    cameras: cameras.length,
    gate,
    pull,
    online,
  });
  const withoutZones = cameras.filter((c) => !(c.zone_count ?? 0));
  return (
    <div className="setup-layout">
      <ol className="setup-steps">
        {SETUP_STEPS.map((s, i) => (
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
        {/* On every later step the download's progress stays in view, so
            nobody wonders on Finish why the button is waiting. */}
        {step !== STEP_AI && (
          <VerifierDownload api={api} mode={mode} compact onStatus={onStatus} />
        )}
        {error && <Notice error>{error}</Notice>}
        {result && <Notice>{result}</Notice>}
        <span className="eyebrow">STEP {step + 1} OF {SETUP_STEPS.length}</span>
        <h2>{SETUP_STEPS[step]}</h2>
        {step === STEP_AI && (
          <>
            <p>
              Argus runs its AI on this computer. The model ({MODEL_SIZE}) is
              downloaded once and reads what each camera sees and checks every
              alert before it reaches you. The download starts now and carries
              on while you set up your cameras; Finish waits for it.
            </p>
            {mode === "demo" ? (
              <Notice>
                Demo mode has no AI model to download. Sample footage only.
              </Notice>
            ) : (
              <VerifierDownload api={api} mode={mode} autoStart onStatus={onStatus} />
            )}
            {mode === "engine" && ready && (
              <Notice>On-device AI ready. Continue to set up your site.</Notice>
            )}
          </>
        )}
        {step === STEP_LOCATIONS && (
          <>
            <p>
              Name the branches and areas of your site first, so a camera has
              somewhere to go when you add it. You can place cameras later.
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
        {step === STEP_CAMERAS && (
          <>
            <p>
              Connect every camera you want watched, one at a time. Each camera
              gets its own scene and, for loitering and intrusion, its own
              zone.
            </p>
            {canConfigureCameras && onAdd && (
              <button className="button primary" onClick={onAdd}>
                <Plus size={16} />
                {cameras.length ? "Add another camera" : "Add camera"}
              </button>
            )}
            {!cameras.length && (
              <Notice>No cameras yet. Add at least one to continue.</Notice>
            )}
          </>
        )}
        {[STEP_CAMERAS, STEP_DETECTORS].includes(step) && (
          <div className="setup-cameras">
            {cameras.map((c) => (
              <div className="setting-row" key={c.id}>
                <div>
                  <strong>{c.id}</strong>
                  <small>
                    {c.area_id || "Ungrouped"}
                    {step === STEP_CAMERAS &&
                      ` · ${c.zone_count ? `${c.zone_count} zone(s)` : "no zone"}`}
                  </small>
                </div>
                {step === STEP_CAMERAS ? (
                  <>
                    <button
                      className="button"
                      onClick={() => onConfigure(c, "scene")}
                    >
                      Review scene
                    </button>
                    <button
                      className="button"
                      onClick={() => onConfigure(c, "zones")}
                    >
                      Draw zone
                    </button>
                  </>
                ) : (
                  <button
                    className="button"
                    onClick={() => onConfigure(c, "detectors")}
                  >
                    Configure detectors
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
        {step === STEP_CAMERAS && cameras.length > 0 && withoutZones.length > 0 && (
          <Notice error>
            Loitering, intrusion and restricted-area alerts only exist inside a
            zone. These cameras have none yet:{" "}
            {withoutZones.map((c) => c.id).join(", ")}. Draw a zone, or open
            the camera and choose "Watch the whole view".
          </Notice>
        )}
        {step === STEP_CAMERAS && cameras.length > 0 && mode === "engine" && !ready && (
          <Notice>
            Scene reviews need the AI model. Cameras added now are mapped
            automatically once the download completes.
          </Notice>
        )}
        {step === STEP_DETECTORS && (
          <>
            <p>
              Pick the use case closest to this site, then tune what each
              camera watches for.
            </p>
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
              Applying a use case changes detector settings for the whole
              site. Zones and English rules stay as you set them.
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
        {step === STEP_ALERTS && (
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
        {step === STEP_FINISH && (
          <>
            <h3>
              {mode === "demo"
                ? "Demo walkthrough complete."
                : "Review before going live."}
            </h3>
            <p>
              {cameras.length} camera(s) configured. Monitoring starts only
              when you choose Start monitoring in the overview.
            </p>
            {mode === "demo" ? (
              <Notice>
                Live readiness checks are unavailable in demo mode. This
                walkthrough does not validate an AI installation.
              </Notice>
            ) : (
              <div className="actions">
                <button
                  className="button"
                  disabled={busy}
                  onClick={() => void run("setup_check")}
                >
                  {busy ? <Spinner /> : <RefreshCw size={16} />}Run readiness
                  checks
                </button>
                <button
                  className="button"
                  disabled={busy}
                  onClick={() => void run("send_test_notification")}
                >
                  Send test notification
                </button>
              </div>
            )}
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
            {mode === "engine" && checks.some((c) => c.ok === false) && (
              <Notice error>
                These still need attention:{" "}
                {checks
                  .filter((c) => c.ok === false)
                  .map((c) => c.label)
                  .join(", ")}
                . You can finish now and fix them from Settings.
              </Notice>
            )}
            {blocked && <Notice error={!ready}>{blocked}</Notice>}
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
          {step < STEP_FINISH ? (
            <button
              className="button primary"
              disabled={busy || (!cameras.length && step === STEP_CAMERAS)}
              title={
                !cameras.length && step === STEP_CAMERAS
                  ? "Add at least one camera first"
                  : undefined
              }
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
              // Readiness checks INFORM; they do not lock the door (Martin,
              // 21 Sep: "finish setup button wasn't clicking", with no word
              // as to why). The one thing Finish does wait for is the AI
              // model -- and the reason is printed right above the button,
              // with the download's progress, for as long as it applies.
              disabled={busy || Boolean(blocked)}
              title={blocked ?? undefined}
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
