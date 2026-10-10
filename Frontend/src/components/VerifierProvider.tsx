import { useCallback, useEffect, useState } from "react";
import { KeyRound, Save, Trash2, Zap } from "lucide-react";
import type { Json, Mode, Transport } from "../lib/types";
import { Notice, Spinner } from "./common";

/**
 * Which AI checks alerts: the bundled on-device model, or a cloud provider
 * with a key. The engine always knew how to talk to several clouds; the app
 * never let the operator pick one, so a five-vCPU server with no GPU spent
 * a minute or more per verdict on the local model (CHI pilot, 8 Oct 2026).
 *
 * The key is write-only here: it is saved on the server in a 0600 file and
 * the form only ever learns whether one is set.
 */

export type VerifierProviderInfo = {
  id: string;
  label: string;
  default_model: string;
  needs_key: boolean;
  local: boolean;
  custom_base_url: boolean;
  note?: string;
};

export type VerifierSettings = {
  provider: string;
  model: string;
  base_url: string;
  label?: string;
  needs_key?: boolean;
  key_set?: boolean;
  local?: boolean;
  providers?: VerifierProviderInfo[];
};

/** The one-line status for the chosen provider. */
export function verifierSummary(s: VerifierSettings | null): string {
  if (!s) return "";
  if (s.local) return "On this computer";
  const key = s.needs_key ? (s.key_set ? "key saved" : "no key yet") : "no key needed";
  return `${s.label || s.provider} · ${s.model || "default model"} · ${key}`;
}

export default function VerifierProvider({
  api,
  mode,
  onSaved,
  notify,
}: {
  api: Transport;
  mode: Mode;
  /** Called after a successful save, so parents re-poll gate status. */
  onSaved?: () => void;
  notify?: (s: string) => void;
}) {
  const [settings, setSettings] = useState<VerifierSettings | null>(null);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState("");

  const load = useCallback(async () => {
    try {
      const s = (await api.invoke<Json>("verifier_settings")) as VerifierSettings;
      setSettings(s);
      setProvider(s.provider);
      setModel(s.model || "");
      setBaseUrl(s.base_url || "");
    } catch (e) {
      setError((e as Error).message);
    }
  }, [api]);

  useEffect(() => {
    if (mode === "demo") return;
    void load();
  }, [load, mode]);

  if (mode === "demo") return null;
  if (!settings) return error ? <Notice error>{error}</Notice> : null;

  const catalogue = settings.providers || [];
  const chosen = catalogue.find((p) => p.id === provider) || catalogue[0];
  const dirtyProvider = provider !== settings.provider;

  function pick(id: string) {
    setProvider(id);
    const next = catalogue.find((p) => p.id === id);
    // A new provider gets its own default model; the same provider keeps
    // whatever the operator typed.
    setModel(id === settings?.provider ? settings.model || "" : next?.default_model || "");
    if (!next?.custom_base_url) setBaseUrl("");
    setResult("");
  }

  async function save() {
    setBusy(true);
    setError("");
    setResult("");
    try {
      const s = (await api.invoke<Json>("set_verifier", [
        provider,
        model,
        baseUrl,
        apiKey,
      ])) as VerifierSettings;
      setSettings(s);
      setApiKey("");
      notify?.("Verifier saved. It applies on the next Start monitoring.");
      onSaved?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function test() {
    setBusy(true);
    setError("");
    setResult("");
    try {
      if (dirtyProvider || apiKey || model !== settings?.model || baseUrl !== settings?.base_url) {
        await save();
      }
      const r = await api.invoke<Json>("test_verifier");
      if (r.ok) setResult(`Works. ${r.detail || ""} ${r.latency_ms ? `(${r.latency_ms} ms)` : ""}`);
      else setError(`Not working: ${r.detail || "no detail"}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function clearKey() {
    setBusy(true);
    setError("");
    try {
      const s = (await api.invoke<Json>("clear_verifier_key")) as VerifierSettings;
      setSettings(s);
      setApiKey("");
      onSaved?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="verifier-provider">
      {error && <Notice error>{error}</Notice>}
      {result && <Notice>{result}</Notice>}
      <label>
        Where alerts are checked
        <select
          id="verifier-provider"
          value={provider}
          onChange={(e) => pick(e.target.value)}
          disabled={busy}
        >
          {catalogue.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </label>
      {chosen?.note && <p className="field-note">{chosen.note}</p>}
      {chosen && !chosen.local && (
        <>
          <label>
            Model
            <input
              id="verifier-model"
              value={model}
              placeholder={chosen.default_model || "model name"}
              onChange={(e) => setModel(e.target.value)}
              disabled={busy}
            />
          </label>
          {chosen.custom_base_url && (
            <label>
              Endpoint base URL
              <input
                id="verifier-base-url"
                value={baseUrl}
                placeholder="https://host/v1"
                onChange={(e) => setBaseUrl(e.target.value)}
                disabled={busy}
              />
            </label>
          )}
          {chosen.needs_key && (
            <label>
              API key
              <input
                id="verifier-api-key"
                type="password"
                autoComplete="off"
                value={apiKey}
                placeholder={
                  settings.key_set && !dirtyProvider
                    ? "A key is saved. Paste a new one to replace it."
                    : "Paste the provider's API key"
                }
                onChange={(e) => setApiKey(e.target.value)}
                disabled={busy}
              />
            </label>
          )}
          <p className="field-note">
            Alert frames, not live video, are sent to this provider over HTTPS.
            Nothing is sent for alerts the rules confirm on their own.
          </p>
        </>
      )}
      <div className="actions">
        <button className="button primary" disabled={busy} onClick={() => void save()}>
          {busy ? <Spinner /> : <Save size={16} />}
          Save verifier
        </button>
        <button className="button" disabled={busy} onClick={() => void test()}>
          <Zap size={16} />
          Test
        </button>
        {chosen && chosen.needs_key && settings.key_set && !dirtyProvider && (
          <button className="button" disabled={busy} onClick={() => void clearKey()}>
            <Trash2 size={16} />
            Remove key
          </button>
        )}
        {chosen && chosen.needs_key && (
          <small className="field-note">
            <KeyRound size={12} /> {settings.key_set && !dirtyProvider ? "Key saved on this computer" : "No key saved"}
          </small>
        )}
      </div>
    </div>
  );
}
