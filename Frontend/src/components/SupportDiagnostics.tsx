import { useState } from "react";
import { Download, Copy } from "lucide-react";
import type { Transport } from "../lib/types";
import { Notice, Spinner } from "./common";

export default function SupportDiagnostics({ api }: { api: Transport }) {
  const [busy, setBusy] = useState(false);
  const [path, setPath] = useState("");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  async function exportBundle() {
    setBusy(true); setError(""); setPath(""); setCopied(false);
    try {
      const result = await api.invoke("download_diagnostics");
      if (!result?.ok || !result.path) throw new Error(result?.error || "Diagnostics export failed");
      setPath(result.path);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return <section className="settings-section">
    <div><h3>Support diagnostics</h3></div>
    <div>
      <button className="button" disabled={busy} onClick={() => void exportBundle()}>
        {busy ? <Spinner /> : <Download size={16} />} Export diagnostics
      </button>
      {error && <Notice error>{error}</Notice>}
      {path && <>
        <label>Diagnostics ZIP on the engine computer<input readOnly value={path} onFocus={e => e.target.select()} /></label>
        <button className="button" onClick={() => {
          void navigator.clipboard.writeText(path).then(() => setCopied(true)).catch(() => setError("Could not copy. Select the file path to copy it manually."));
        }}><Copy size={16} />{copied ? "Copied" : "Copy path"}</button>
      </>}
    </div>
  </section>;
}
