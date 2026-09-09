import { useEffect, useState } from "react";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Plus,
  RefreshCw,
} from "lucide-react";
import type { Camera, Json, Mode, Transport } from "../lib/types";
import { Badge, Notice, Spinner } from "./common";
const steps = [
  "Cameras",
  "Scenes & zones",
  "Use case",
  "Detectors",
  "Verification",
  "Finish",
];
export default function Setup({
  api,
  mode,
  cameras,
  onAdd,
  onConfigure,
  onChange,
  onFinish,
}: {
  api: Transport;
  mode: Mode;
  cameras: Camera[];
  onAdd: () => void;
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
        {error && <Notice error>{error}</Notice>}
        {result && <Notice>{result}</Notice>}
        <span className="eyebrow">STEP {step + 1} OF 6</span>
        <h2>{steps[step]}</h2>
        {step === 0 && (
          <>
            <p>Connect cameras and assign them to the areas of your site.</p>
            <button className="button primary" onClick={onAdd}>
              <Plus size={16} />
              Add camera
            </button>
          </>
        )}
        {[0, 1, 3].includes(step) && (
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
                    onConfigure(c, step === 3 ? "detectors" : "scene")
                  }
                >
                  {step === 3 ? "Configure detectors" : "Review scene"}
                </button>
                {step === 1 && (
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
        {step === 1 && (
          <Notice>
            Each camera needs its own evidence. Review the scene and correct it
            before approval, even when cameras share an area.
          </Notice>
        )}
        {step === 2 && (
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
        {step === 4 && (
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
        {step === 5 && (
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
              disabled={busy || (!cameras.length && step === 0)}
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
              disabled={
                busy ||
                !cameras.length ||
                (mode === "engine" &&
                  (!checks.length || checks.some((c) => c.ok === false)))
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
