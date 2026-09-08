import { useEffect, useState } from "react";
import type { Json, Transport } from "../lib/types";
import { Notice, Spinner } from "./common";
export default function FeedSwitcher({
  api,
  onChange,
  disabled,
}: {
  api: Transport;
  onChange: () => Promise<void>;
  disabled: boolean;
}) {
  const [feeds, setFeeds] = useState<Json>({ sources: [] });
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api
      .invoke("feed_sources")
      .then((r) => active && setFeeds(r))
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [api]);
  useEffect(() => {
    if (!busy) return;
    let active = true,
      inFlight = false;
    const timer = setInterval(async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const r = await api.invoke("feed_switch_status");
        if (!active) return;
        setStatus(r.status || "Switching source");
        if (!r.busy) {
          setBusy(false);
          if (r.error) setError(r.error);
          else {
            setFeeds(await api.invoke("feed_sources"));
            await onChange();
            setStatus("");
          }
        }
      } catch (e) {
        if (active) {
          setError((e as Error).message);
          setBusy(false);
        }
      } finally {
        inFlight = false;
      }
    }, 1500);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [api, busy, onChange]);
  async function change(key: string) {
    setError("");
    setBusy(true);
    try {
      await api.invoke("switch_feed", [key]);
      setStatus("Switching source");
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  }
  return (
    <>
      {error && <Notice error>{error}</Notice>}
      <div className="feed-controls">
        <label>
          Camera source
          <select
            aria-label="Camera source group"
            value={feeds.active || ""}
            disabled={disabled || busy}
            onChange={(e) => void change(e.target.value)}
          >
            <option value="" disabled>
              Current site configuration
            </option>
            {feeds.sources.map((s: Json) => (
              <option key={s.key} value={s.key}>
                {s.label}
                {s.kind === "demo" ? " (recorded footage)" : ""}
              </option>
            ))}
          </select>
        </label>
        {busy && (
          <span>
            <Spinner />
            {status}
          </span>
        )}
      </div>
    </>
  );
}
