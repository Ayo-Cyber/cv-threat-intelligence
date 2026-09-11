import { useState } from "react";
import { Plus, Search, CheckCircle2 } from "lucide-react";
import type { Hierarchy, Json, Mode, Transport } from "../lib/types";
import { Notice, Spinner } from "./common";

const ASSIGN_LATER = "assign-later";

export default function AddCamera({
  api,
  mode,
  hierarchy,
  onAdded,
}: {
  api: Transport;
  mode: Mode;
  hierarchy: Hierarchy;
  onAdded: () => Promise<void>;
}) {
  const [id, setId] = useState("");
  const [source, setSource] = useState("");
  const [branch, setBranch] = useState("");
  const [area, setArea] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState("");
  const [found, setFound] = useState<Json[]>([]);
  const selectedBranch = hierarchy.branches.find((item) => item.id === branch);
  const placementReady =
    branch === ASSIGN_LATER || Boolean(selectedBranch && area);
  async function action(kind: string) {
    setBusy(true);
    setError("");
    setResult("");
    try {
      if (kind === "discover") {
        const r = await api.invoke("discover_cameras");
        setFound(r.cameras || []);
        setResult(`${r.count || 0} cameras discovered`);
      } else if (kind === "test") {
        const r = await api.invoke("test", [source]);
        if (r.ok === false)
          throw new Error(r.message || "Camera connection failed");
        setResult("Camera source is reachable");
      } else {
        await api.invoke("add_camera", [
          {
            id: id.trim(),
            source: source.trim(),
            area_id: branch === ASSIGN_LATER ? undefined : area,
          },
        ]);
        await onAdded();
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      {mode === "demo" && (
        <Notice>
          Camera discovery and connection testing require local engine mode.
          Demo camera additions are saved only to this browser.
        </Notice>
      )}
      {error && <Notice error>{error}</Notice>}
      <button
        className="button"
        disabled={busy || mode === "demo"}
        onClick={() => void action("discover")}
      >
        <Search size={16} />
        Discover network cameras
      </button>
      {result && (
        <p className="success">
          <CheckCircle2 size={16} />
          {result}
        </p>
      )}
      {found.map((c, i) => (
        <button
          className="discovery-row"
          key={i}
          onClick={() => {
            setSource(c.rtsp_url || c.url || "");
            setId(c.name || c.ip || "");
          }}
        >
          {c.name || c.ip || c.host || `Camera ${i + 1}`}
          <span>{c.rtsp_url || c.url || c.address}</span>
        </button>
      ))}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void action("add");
        }}
      >
        <label>
          Camera name
          <input
            required
            pattern="[a-zA-Z0-9_-]+"
            title="Use letters, numbers, underscores or hyphens"
            value={id}
            onChange={(e) => setId(e.target.value)}
            placeholder="reception_01"
          />
        </label>
        <label>
          Camera source
          <input
            required
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="0 for webcam, RTSP URL or local video path"
            autoComplete="off"
          />
        </label>
        <label>
          Branch
          <select
            required
            value={branch}
            onChange={(event) => {
              setBranch(event.target.value);
              setArea("");
            }}
          >
            <option value="">Select a branch</option>
            {hierarchy.branches.map((item) => (
              <option value={item.id} key={item.id}>
                {item.name}
              </option>
            ))}
            <option value={ASSIGN_LATER}>Assign later</option>
          </select>
        </label>
        {selectedBranch && (
          <label>
            Area
            <select
              required
              value={area}
              onChange={(event) => setArea(event.target.value)}
            >
              <option value="">Select an area</option>
              {selectedBranch.areas.map((item) => (
                <option value={item.id} key={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
        )}
        {branch === ASSIGN_LATER && (
          <p className="field-note">
            This camera will remain visible under Unassigned until it is placed
            in an area.
          </p>
        )}
        <div className="actions">
          <button
            className="button"
            type="button"
            disabled={busy || !source || mode === "demo"}
            onClick={() => void action("test")}
          >
            Test connection
          </button>
          <button
            className="button primary"
            disabled={busy || !id || !source || !placementReady}
          >
            {busy ? <Spinner /> : <Plus size={16} />}Add camera
          </button>
        </div>
      </form>
    </>
  );
}
