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
}: {
  api: Transport;
  mode: Mode;
  autoStart?: boolean;
}) {
  const [gate, setGate] = useState<GateStatus | null>(null);
  const [pull, setPull] = useState<PullProgress | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const started = useRef(false);

  const poll = useCallback(async () => {
    try {
      const status = await api.invoke<Json>("gate_status");
      setGate(status as GateStatus);
      const progress = await api.invoke<Json>("pull_progress");
      setPull(progress as PullProgress);
      return { status, progress } as {
        status: GateStatus;
        progress: PullProgress;
      };
    } catch (e) {
      setError((e as Error).message);
      return null;
    }
  }, [api]);

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
  return (
    <div className="verifier-download">
      {error && <Notice error>{error}</Notice>}
      <Notice error={message.tone === "action"}>{message.text}</Notice>
      {pulling && (
        <progress className="verifier-progress" max={100} value={percent} />
      )}
      {message.tone === "action" && (
        <div className="actions">
          <button className="button primary" disabled={busy} onClick={() => void download()}>
            {busy ? <Spinner /> : <Download size={16} />}
            Download model ({MODEL_SIZE})
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
