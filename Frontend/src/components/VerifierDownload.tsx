import { useCallback, useEffect, useRef, useState } from "react";
import { Download, RefreshCw } from "lucide-react";
import type { Json, Mode, Transport } from "../lib/types";
import { Notice, Spinner } from "./common";

/**
 * The on-device AI model: ~3.3 GB, fetched once, not shipped in the installer.
 *
 * The retired console started this pull at step 0 of setup so it ran WHILE the
 * operator worked ("3.3 GB downloads WHILE they set up, not after") and showed
 * progress throughout. The React wizard shipped without any of it: no
 * auto-start, no progress, and no button — while setup_check still told people
 * to "Download the model in the Verification step", naming a control that no
 * longer existed. A pilot installed v1.8.15 and reported he never saw the
 * 3.3 GB download, because nothing ever started one (Martin, 20 Sep).
 *
 * Ollama resumes partial downloads, so an interrupted pull continues.
 */

export const MODEL_SIZE = "3.3 GB";

export type GateStatus = {
  mode?: string;
  model?: string;
  runtime_bundled?: boolean;
};

export type PullProgress = {
  state?: string;
  percent?: number;
  detail?: string;
};

/** The object-recognition model (SigLIP), installed by the same Download AI
 * click. It matches camera crops against a customer's reference photos; the
 * verification model does not do that job, and downloading one never
 * installs the other (Demi's handoff, 5 Oct). Finish does NOT wait for it:
 * a site that never enrols a product loses nothing, and a failure here
 * disables object recognition only. */
export const RECOGNITION_SIZE = "0.8 GB";

export type RecognitionStatus = {
  state?: string;
  percent?: number;
  detail?: string;
  display_size?: string;
};

export function recognitionMessage(
  status: RecognitionStatus | null,
): { tone: "ready" | "working" | "action"; text: string } | null {
  if (!status) return null;
  const size = status.display_size || RECOGNITION_SIZE;
  const percent = Math.max(0, Math.min(100, Math.round(status.percent ?? 0)));
  switch (status.state) {
    case "ready":
      return { tone: "ready", text: "Object-recognition model ready." };
    case "downloading":
      return {
        tone: "working",
        text: `Downloading the object-recognition model (${size}) — ${percent}%.`,
      };
    case "verifying":
      return {
        tone: "working",
        text: "Checking the object-recognition model: loading it once and asking for an embedding.",
      };
    case "error":
      return {
        tone: "action",
        text: `The object-recognition model did not install: ${status.detail || "unknown error"}. Object recognition stays off until it does; everything else works.`,
      };
    default:
      return {
        tone: "action",
        text: `The object-recognition model (${size}) has not been installed yet. Object recognition stays off until it is.`,
      };
  }
}

/** What the operator should be told, given gate status and pull progress. */
export function verifierMessage(
  gate: GateStatus | null,
  pull: PullProgress | null,
): { tone: "ready" | "working" | "action"; text: string } | null {
  if (!gate) return null;
  if (gate.mode === "live")
    return { tone: "ready", text: "On-device AI ready." };
  if (pull?.state === "pulling") {
    const percent = Math.max(0, Math.min(100, Math.round(pull.percent ?? 0)));
    return {
      tone: "working",
      text: `Downloading the on-device AI (${MODEL_SIZE}) — ${percent}%. You can carry on setting up.`,
    };
  }
  if (pull?.state === "error")
    return {
      tone: "action",
      text: `The download stopped: ${pull.detail || "unknown error"}. It resumes where it left off.`,
    };
  if (gate.mode === "no-model")
    return {
      tone: "action",
      text: `The on-device AI model (${MODEL_SIZE}) has not been downloaded yet. Alerts stay unverified until it is.`,
    };
  return {
    tone: "action",
    text: "The on-device AI runtime is not running yet.",
  };
}

export default function VerifierDownload({
  api,
  mode,
  autoStart = false,
  onStatus,
  compact = false,
}: {
  api: Transport;
  mode: Mode;
  autoStart?: boolean;
  /** Every poll's result, so the wizard can gate Finish on the model. */
  onStatus?: (gate: GateStatus | null, pull: PullProgress | null) => void;
  /** One line of status, no buttons: the banner shown on later steps. */
  compact?: boolean;
}) {
  const [gate, setGate] = useState<GateStatus | null>(null);
  const [pull, setPull] = useState<PullProgress | null>(null);
  const [recognition, setRecognition] = useState<RecognitionStatus | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const started = useRef(false);

  const poll = useCallback(async () => {
    try {
      const status = await api.invoke<Json>("gate_status");
      setGate(status as GateStatus);
      const progress = await api.invoke<Json>("pull_progress");
      setPull(progress as PullProgress);
      onStatus?.(status as GateStatus, progress as PullProgress);
      try {
        setRecognition((await api.invoke<Json>("recognition_model_status")) as RecognitionStatus);
      } catch {
        // An older engine without the route: the verification model's status
        // still shows; the recognition line simply stays absent.
        setRecognition(null);
      }
      return { status, progress } as {
        status: GateStatus;
        progress: PullProgress;
      };
    } catch (e) {
      setError((e as Error).message);
      return null;
    }
  }, [api, onStatus]);

  const download = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      await api.invoke("pull_model");
      await poll();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [api, poll]);

  const retryRecognition = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      await api.invoke("pull_recognition_model");
      await poll();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [api, poll]);

  useEffect(() => {
    if (mode === "demo") return;
    let live = true;
    const tick = async () => {
      const seen = await poll();
      if (!live || !seen) return;
      // The old console kicked the pull off at step 0 so the slowest part of
      // setup overlapped the rest of it. Only ever once per mount.
      if (
        autoStart &&
        !started.current &&
        seen.status.mode === "no-model" &&
        seen.progress.state !== "pulling"
      ) {
        started.current = true;
        void download();
      }
    };
    void tick();
    const timer = setInterval(() => void tick(), 4000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [autoStart, download, mode, poll]);

  if (mode === "demo") return null;
  const message = verifierMessage(gate, pull);
  if (!message) return null;
  const pulling = pull?.state === "pulling";
  const percent = Math.max(0, Math.min(100, Math.round(pull?.percent ?? 0)));
  const offline =
    typeof navigator !== "undefined" && navigator.onLine === false;
  const recognitionLine = recognitionMessage(recognition);
  if (compact && message.tone === "ready" && (!recognitionLine || recognitionLine.tone === "ready"))
    return null;
  return (
    <div className="verifier-download">
      {error && <Notice error>{error}</Notice>}
      {!(compact && message.tone === "ready") && (
        <Notice error={message.tone === "action"}>{message.text}</Notice>
      )}
      {pulling && (
        <progress className="verifier-progress" max={100} value={percent} />
      )}
      {recognitionLine && !(compact && recognitionLine.tone === "ready") && (
        <div className="recognition-download">
          <Notice error={recognitionLine.tone === "action" && recognition?.state === "error"}>
            {recognitionLine.text}
          </Notice>
          {recognition?.state === "downloading" && (
            <progress
              className="verifier-progress"
              max={100}
              value={Math.max(0, Math.min(100, Math.round(recognition.percent ?? 0)))}
            />
          )}
          {!compact && recognitionLine.tone === "action" && (
            <div className="actions">
              <button className="button" disabled={busy} onClick={() => void retryRecognition()}>
                {busy ? <Spinner /> : <Download size={16} />}
                {recognition?.state === "error" ? "Retry" : "Install"} object-recognition model ({RECOGNITION_SIZE})
              </button>
            </div>
          )}
        </div>
      )}
      {offline && message.tone !== "ready" && (
        <Notice error>
          This computer is offline. The model ({MODEL_SIZE}) needs an internet
          connection; the download resumes where it stopped once you are back
          online.
        </Notice>
      )}
      {!compact && message.tone === "action" && (
        <div className="actions">
          <button className="button primary" disabled={busy} onClick={() => void download()}>
            {busy ? <Spinner /> : <Download size={16} />}
            Download model ({MODEL_SIZE} + {RECOGNITION_SIZE} recognition)
          </button>
          <button className="button" disabled={busy} onClick={() => void poll()}>
            <RefreshCw size={16} />
            Recheck
          </button>
        </div>
      )}
    </div>
  );
}
