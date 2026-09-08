import { useEffect, useState } from "react";
import { RefreshCw, ShieldCheck } from "lucide-react";
import type { Json, Transport } from "../lib/types";
import { Badge, Notice, Spinner } from "./common";
export default function HierarchyReview({
  api,
  onCamera,
  onChange,
}: {
  api: Transport;
  onCamera: (id: string) => void;
  onChange: () => Promise<void>;
}) {
  const [summary, setSummary] = useState<Json | null>(null);
  const [area, setArea] = useState("");
  const [draft, setDraft] = useState<Json | null>(null);
  const [site, setSite] = useState<Json | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function load() {
    const r = await api.invoke("scene_review_summary");
    setSummary(r);
    setSite(r.site);
    const a = r.areas.find((a: Json) => a.id === area) || r.areas[0];
    setArea(a?.id || "");
    setDraft(a?.context || null);
  }
  useEffect(() => {
    let active = true;
    api
      .invoke("scene_review_summary")
      .then((r) => {
        if (active) {
          setSummary(r);
          setSite(r.site);
          setArea(r.areas[0]?.id || "");
          setDraft(r.areas[0]?.context || null);
        }
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, [api]);
  async function act(method: string, args: unknown[]) {
    setBusy(true);
    setError("");
    try {
      await api.invoke(method, args);
      await load();
      await onChange();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const current = summary?.areas.find((a: Json) => a.id === area);
  return (
    <>
      {error && <Notice error>{error}</Notice>}
      {!summary ? (
        <Spinner />
      ) : (
        <>
          <div className="context-status">
            <Badge tone="green">{summary.counts.reviewed} reviewed</Badge>
            <span>
              {summary.counts.total_cameras} cameras ·{" "}
              {summary.mapping.pending || 0} pending ·{" "}
              {summary.mapping.failed || 0} failed
            </span>
            <button
              className="icon-button"
              title="Reload scene review"
              onClick={() => void load().catch((e) => setError(e.message))}
            >
              <RefreshCw size={16} />
            </button>
          </div>
          <h3>Site context</h3>
          {site ? (
            <>
              <label>
                Site type
                <input
                  value={site.site_type}
                  onChange={(e) =>
                    setSite({ ...site, site_type: e.target.value })
                  }
                />
              </label>
              <label>
                Site description
                <textarea
                  value={site.site_description}
                  onChange={(e) =>
                    setSite({ ...site, site_description: e.target.value })
                  }
                />
              </label>
              <button
                className="button"
                disabled={busy}
                onClick={() => void act("approve_site_context", [site])}
              >
                <ShieldCheck size={16} />
                Approve site context
              </button>
            </>
          ) : (
            <Notice>
              No site-level proposal exists yet. Review individual camera scenes
              while mapping completes.
            </Notice>
          )}
          <div className="divider" />
          <label>
            Area
            <select
              value={area}
              onChange={(e) => {
                setArea(e.target.value);
                setDraft(
                  summary.areas.find((a: Json) => a.id === e.target.value)
                    ?.context || null,
                );
              }}
            >
              {summary.areas.map((a: Json) => (
                <option key={a.id} value={a.id}>
                  {a.name || a.id}
                </option>
              ))}
            </select>
          </label>
          <div className="hierarchy-cameras">
            {current?.cameras.map((c: Json) => (
              <button key={c.camera_id} onClick={() => onCamera(c.camera_id)}>
                {c.source_frame_uri ? (
                  <img
                    src={c.source_frame_uri}
                    alt={`${c.camera_id} mapping evidence`}
                  />
                ) : (
                  <span className="no-evidence">No evidence frame</span>
                )}
                <strong>{c.camera_id}</strong>
                <small>{c.mapping.status.replaceAll("_", " ")}</small>
              </button>
            ))}
          </div>
          {draft ? (
            <>
              <label>
                Area type
                <input
                  value={draft.area_type}
                  onChange={(e) =>
                    setDraft({ ...draft, area_type: e.target.value })
                  }
                />
              </label>
              <label>
                Area description
                <textarea
                  value={draft.area_description}
                  onChange={(e) =>
                    setDraft({ ...draft, area_description: e.target.value })
                  }
                />
              </label>
              {!current?.bulk_reviewable && (
                <Notice error>
                  Area approval is blocked: camera evidence is incomplete or the
                  viewpoints conflict. Review each camera first.
                </Notice>
              )}
              <button
                className="button primary"
                disabled={busy || !current?.bulk_reviewable}
                onClick={() => void act("approve_area_context", [area, draft])}
              >
                <ShieldCheck size={16} />
                Approve this area
              </button>
            </>
          ) : (
            <Notice>
              No area-level proposal exists yet. Each camera can still be
              reviewed individually.
            </Notice>
          )}
        </>
      )}
    </>
  );
}
