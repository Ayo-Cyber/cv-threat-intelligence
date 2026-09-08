import { useEffect, useRef, useState } from "react";
import { Undo2, Trash2, MousePointer2, Check, Square } from "lucide-react";
import type { Camera, Point, Transport, Zone } from "../lib/types";
import { relativePoint, validPolygon } from "../lib/geometry";
import { Notice, Spinner } from "./common";
export default function ZoneEditor({
  camera,
  api,
  onSaved,
}: {
  camera: Camera;
  api: Transport;
  onSaved: () => void;
}) {
  const [image, setImage] = useState("");
  const [size, setSize] = useState({ width: 640, height: 480 });
  const [points, setPoints] = useState<Point[]>([]);
  const [zones, setZones] = useState<Zone[]>([]);
  const [name, setName] = useState("");
  const [dwell, setDwell] = useState(5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [tool, setTool] = useState<"polygon" | "rectangle">("polygon");
  const start = useRef<Point | null>(null);
  useEffect(() => {
    let active = true;
    setBusy(true);
    Promise.all([
      api.invoke("camera_snapshot", [camera.id]),
      api.invoke<Zone[]>("list_zones", [camera.id]),
    ])
      .then(([s, z]) => {
        if (active) {
          setImage(s.uri || camera.snapshot || "");
          setSize({ width: s.w || 640, height: s.h || 480 });
          setZones(z);
        }
      })
      .catch((e) => active && setError(e.message))
      .finally(() => active && setBusy(false));
    return () => {
      active = false;
    };
  }, [api, camera.id, camera.snapshot]);
  async function save() {
    if (!validPolygon(points) || !name.trim() || dwell < 1) return;
    setBusy(true);
    setError("");
    try {
      await api.invoke("add_zone", [camera.id, name.trim(), points, dwell]);
      setZones(await api.invoke("list_zones", [camera.id]));
      setPoints([]);
      setName("");
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="zone-editor">
      <div className="section-heading">
        <div>
          <h3>Zones of interest</h3>
          <p>Loitering alerts use the time a person spends inside a zone.</p>
        </div>
      </div>
      {error && <Notice error>{error}</Notice>}
      <div className="zone-tools">
        <div className="segmented">
          <button
            className={tool === "polygon" ? "active" : ""}
            title="Polygon"
            aria-label="Draw polygon"
            onClick={() => {
              setTool("polygon");
              setPoints([]);
            }}
          >
            <MousePointer2 size={16} />
          </button>
          <button
            className={tool === "rectangle" ? "active" : ""}
            title="Rectangle"
            aria-label="Draw rectangle"
            onClick={() => {
              setTool("rectangle");
              setPoints([]);
            }}
          >
            <Square size={16} />
          </button>
        </div>
        <button
          className="icon-button"
          title="Undo last point"
          aria-label="Undo last point"
          onClick={() => setPoints((p) => p.slice(0, -1))}
        >
          <Undo2 size={17} />
        </button>
        <button
          className="icon-button"
          title="Clear drawing"
          aria-label="Clear drawing"
          onClick={() => setPoints([])}
        >
          <Trash2 size={16} />
        </button>
        <span>{points.length} points</span>
      </div>
      {busy && !image ? (
        <Spinner />
      ) : image ? (
        <div
          className="zone-canvas"
          style={{ aspectRatio: `${size.width}/${size.height}` }}
        >
          <img
            src={image}
            alt={`${camera.id} zone reference`}
            onLoad={(e) => {
              const img = e.currentTarget;
              setSize({ width: img.naturalWidth, height: img.naturalHeight });
            }}
          />
          <svg
            viewBox={`0 0 ${size.width} ${size.height}`}
            role="img"
            aria-label="Zone drawing canvas"
            onPointerDown={(e) => {
              const p = relativePoint(
                e.clientX,
                e.clientY,
                e.currentTarget.getBoundingClientRect(),
                size.width,
                size.height,
              );
              if (tool === "polygon")
                setPoints((old) => (old.length < 30 ? [...old, p] : old));
              else {
                start.current = p;
                e.currentTarget.setPointerCapture(e.pointerId);
                setPoints([p, p, p, p]);
              }
            }}
            onPointerMove={(e) => {
              if (tool !== "rectangle" || !start.current) return;
              const p = relativePoint(
                  e.clientX,
                  e.clientY,
                  e.currentTarget.getBoundingClientRect(),
                  size.width,
                  size.height,
                ),
                s = start.current;
              setPoints([s, [p[0], s[1]], p, [s[0], p[1]]]);
            }}
            onPointerUp={() => {
              start.current = null;
            }}
          >
            {zones.map((z) => (
              <g key={z.name}>
                <polygon
                  points={z.points.map((p) => p.join(",")).join(" ")}
                  className="saved-zone"
                />
                <text x={z.points[0]?.[0] + 5} y={z.points[0]?.[1] + 18}>
                  {z.name}
                </text>
              </g>
            ))}
            <polygon
              points={points.map((p) => p.join(",")).join(" ")}
              className="draft-zone"
            />
            {points.map((p, i) => (
              <circle key={i} cx={p[0]} cy={p[1]} r={4} />
            ))}
          </svg>
        </div>
      ) : (
        <Notice>
          Camera evidence is unavailable. Connect the source before drawing a
          zone.
        </Notice>
      )}
      <div className="form-row">
        <label>
          Zone name
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Reception"
          />
        </label>
        <label>
          Dwell threshold (seconds)
          <input
            type="number"
            min="1"
            max="86400"
            value={dwell}
            onChange={(e) => setDwell(Number(e.target.value))}
          />
        </label>
      </div>
      {points.length >= 3 && !validPolygon(points) && (
        <Notice>
          The zone must have a non-zero area and no crossing edges.
        </Notice>
      )}
      <button
        className="button primary"
        disabled={
          busy ||
          !image ||
          !validPolygon(points) ||
          !name.trim() ||
          !Number.isFinite(dwell) ||
          dwell < 1 ||
          dwell > 86400
        }
        onClick={save}
      >
        <Check size={16} />
        Save zone
      </button>
      <div className="zone-list">
        {zones.map((z) => (
          <div key={z.name}>
            <div>
              <strong>{z.name}</strong>
              <small>
                Loitering after {z.dwell_alert_seconds}s · {z.points.length}{" "}
                points
              </small>
            </div>
            <button
              className="icon-button"
              aria-label={`Remove ${z.name}`}
              title="Remove zone"
              onClick={async () => {
                try {
                  await api.invoke("remove_zone", [camera.id, z.name]);
                  setZones(await api.invoke("list_zones", [camera.id]));
                  onSaved();
                } catch (e) {
                  setError((e as Error).message);
                }
              }}
            >
              <Trash2 size={16} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
