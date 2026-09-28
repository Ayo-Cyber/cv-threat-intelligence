import { useEffect, useRef, useState } from "react";

/** Four corners of the camera's own frame, in its ORIGINAL pixels -- what
 * add_zone expects. Loitering and intrusion only exist inside a zone; this
 * gives an operator one without drawing a polygon. */
export function wholeViewPolygon(size: { width: number; height: number }) {
  const w = Math.max(1, Math.round(size.width));
  const h = Math.max(1, Math.round(size.height));
  return [[0, 0], [w, 0], [w, h], [0, h]];
}

import {
  Undo2,
  Trash2,
  MousePointer2,
  Check,
  Square,
  Milestone,
  ArrowLeftRight,
} from "lucide-react";
import type { Camera, Point, Transport, VehicleLine, Zone } from "../lib/types";
import { relativePoint, validPolygon } from "../lib/geometry";
import {
  enterArrow,
  enterDirectionLabel,
  toFraction,
  toPixels,
  validLine,
} from "../lib/gate-line";
import { Notice, Spinner } from "./common";
export default function ZoneEditor({
  camera,
  api,
  onSaved,
  onDirtyChange,
}: {
  camera: Camera;
  api: Transport;
  onSaved: () => void;
  onDirtyChange?: (dirty: boolean) => void;
}) {
  const [image, setImage] = useState("");
  const [size, setSize] = useState({ width: 640, height: 480 });
  const [points, setPoints] = useState<Point[]>([]);
  const [zones, setZones] = useState<Zone[]>([]);
  const [name, setName] = useState("");
  const [dwell, setDwell] = useState(5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [tool, setTool] = useState<"polygon" | "rectangle" | "line">(
    "rectangle",
  );
  // The vehicle tripwire (KPI 2). One per camera; `line` is what is saved,
  // `draft` the two clicks being placed, `flip` which side is "entering".
  const [line, setLine] = useState<VehicleLine | null>(null);
  const [draft, setDraft] = useState<Point[]>([]);
  const [flip, setFlip] = useState(false);
  const [lineName, setLineName] = useState("gate");
  const stage = useRef<HTMLDivElement>(null);
  const [available, setAvailable] = useState({ width: 640, height: 480 });
  useEffect(() => {
    onDirtyChange?.(
      points.length > 0 || Boolean(name.trim()) || draft.length > 0,
    );
  }, [points, name, draft, onDirtyChange]);
  useEffect(() => {
    if (!stage.current) return;
    const observer = new ResizeObserver(([entry]) => {
      setAvailable({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      });
    });
    observer.observe(stage.current);
    return () => observer.disconnect();
  }, []);
  const scale = Math.min(
    available.width / size.width,
    available.height / size.height,
  );
  const start = useRef<Point | null>(null);
  useEffect(() => {
    let active = true;
    setBusy(true);
    Promise.all([
      api.invoke("camera_snapshot", [camera.id]),
      api.invoke<Zone[]>("list_zones", [camera.id]),
      // Older engines have no tripwire route: treat that as "no line" rather
      // than blocking the zones the operator came here to draw.
      api
        .invoke<{ vehicle_line?: VehicleLine | null }>("vehicle_line", [
          camera.id,
        ])
        .catch(() => ({ vehicle_line: null })),
    ])
      .then(([s, z, l]) => {
        if (active) {
          setImage(s.uri || camera.snapshot || "");
          setSize({ width: s.w || 640, height: s.h || 480 });
          setZones(z);
          const saved = l?.vehicle_line ?? null;
          setLine(saved);
          if (saved) {
            setFlip(Boolean(saved.flip));
            setLineName(saved.name || "gate");
          }
        }
      })
      .catch((e) => active && setError(e.message))
      .finally(() => active && setBusy(false));
    return () => {
      active = false;
    };
  }, [api, camera.id, camera.snapshot]);
  async function watchWholeView() {
    setBusy(true);
    setError("");
    try {
      await api.invoke("add_zone", [camera.id, "whole view", wholeViewPolygon(size), dwell]);
      setZones(await api.invoke("list_zones", [camera.id]));
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

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
  async function saveLine() {
    if (!validLine(draft[0], draft[1])) return;
    setBusy(true);
    setError("");
    try {
      const r = await api.invoke<{ vehicle_line?: VehicleLine }>(
        "set_vehicle_line",
        [
          camera.id,
          toFraction(draft[0], size),
          toFraction(draft[1], size),
          flip,
          lineName.trim() || "gate",
        ],
      );
      setLine(
        r?.vehicle_line ??
          (await api.invoke<{ vehicle_line?: VehicleLine | null }>(
            "vehicle_line",
            [camera.id],
          ))?.vehicle_line ??
          null,
      );
      setDraft([]);
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function removeLine() {
    setBusy(true);
    setError("");
    try {
      await api.invoke("remove_vehicle_line", [camera.id]);
      setLine(null);
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  // Pixels of the frame being shown, for the saved line and the draft.
  const shown = line ? toPixels(line, size) : null;
  const draftLine = validLine(draft[0], draft[1])
    ? { start: draft[0], end: draft[1] }
    : null;
  return (
    <div className="zone-editor">
      <div className="zone-workspace">
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
                setDraft([]);
              }}
            >
              <Square size={16} />
            </button>
            <button
              className={tool === "line" ? "active" : ""}
              title="Vehicle gate line: click where the line starts, then where it ends"
              aria-label="Draw vehicle gate line"
              onClick={() => {
                setTool("line");
                setPoints([]);
                setDraft([]);
              }}
            >
              <Milestone size={16} />
            </button>
          </div>
          <button
            className="icon-button"
            title="Undo last point"
            aria-label="Undo last point"
            onClick={() =>
              tool === "line"
                ? setDraft((p) => p.slice(0, -1))
                : setPoints((p) => p.slice(0, -1))
            }
          >
            <Undo2 size={17} />
          </button>
          <button
            className="icon-button"
            title="Clear drawing"
            aria-label="Clear drawing"
            onClick={() => (tool === "line" ? setDraft([]) : setPoints([]))}
          >
            <Trash2 size={16} />
          </button>
          <span>
            {tool === "line"
              ? draft.length === 0
                ? "click the line's start"
                : draft.length === 1
                  ? "click the line's end"
                  : "line placed"
              : `${points.length} points`}
          </span>
        </div>
        <div className="zone-stage" ref={stage}>
          {busy && !image ? (
            <Spinner />
          ) : image ? (
            <div
              className="zone-canvas"
              style={{ width: size.width * scale, height: size.height * scale }}
            >
              <img
                src={image}
                alt={`${camera.id} zone reference`}
                onLoad={(e) => {
                  const img = e.currentTarget;
                  setSize({
                    width: img.naturalWidth,
                    height: img.naturalHeight,
                  });
                }}
              />
              <svg
                viewBox={`0 0 ${size.width} ${size.height}`}
                role="img"
                aria-label="Zone drawing canvas"
                onPointerDown={(e) => {
                  if (e.button !== 0 || busy) return;
                  const p = relativePoint(
                    e.clientX,
                    e.clientY,
                    e.currentTarget.getBoundingClientRect(),
                    size.width,
                    size.height,
                  );
                  if (tool === "line") {
                    setDraft((old) => (old.length >= 2 ? [p] : [...old, p]));
                    return;
                  }
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
                onPointerCancel={() => {
                  start.current = null;
                }}
                onLostPointerCapture={() => {
                  start.current = null;
                }}
              >
                {zones.map((z) => (
                  <g key={z.name}>
                    <polygon
                      points={z.polygon.map((p) => p.join(",")).join(" ")}
                      className="saved-zone"
                    />
                    <text x={z.polygon[0]?.[0] + 5} y={z.polygon[0]?.[1] + 18}>
                      {z.name}
                    </text>
                  </g>
                ))}
                {shown && (
                  <GateLine
                    start={shown.start}
                    end={shown.end}
                    flip={Boolean(line?.flip)}
                    label={line?.name || "gate"}
                    className="saved-line"
                    reach={Math.max(24, size.width * 0.06)}
                  />
                )}
                {draftLine && (
                  <GateLine
                    start={draftLine.start}
                    end={draftLine.end}
                    flip={flip}
                    label={lineName.trim() || "gate"}
                    className="draft-line"
                    reach={Math.max(24, size.width * 0.06)}
                  />
                )}
                {draft.map((p, i) => (
                  <circle key={`d${i}`} cx={p[0]} cy={p[1]} r={4} />
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
              Camera evidence is unavailable. Connect the source before drawing
              a zone.
            </Notice>
          )}
        </div>
      </div>
      <aside className="zone-properties">
        {tool === "line" ? (
          <>
            <h3>Vehicle gate line</h3>
            <p className="muted">
              A vehicle whose centre crosses this line in the arrow's
              direction is ENTERING; the other way is EXITING. One line per
              camera.
            </p>
            <div className="form-row">
              <label>
                Line name
                <input
                  value={lineName}
                  onChange={(e) => setLineName(e.target.value)}
                  placeholder="gate"
                />
              </label>
            </div>
            <button
              type="button"
              className="button"
              disabled={busy || !draftLine}
              title="Swap which side of the line counts as entering"
              onClick={() => setFlip((f) => !f)}
            >
              <ArrowLeftRight size={16} />
              Swap direction
              {draftLine &&
                ` (entering ${enterDirectionLabel(draftLine.start, draftLine.end, flip)})`}
            </button>
            <button
              className="button primary"
              disabled={busy || !image || !draftLine || !lineName.trim()}
              onClick={() => void saveLine()}
            >
              <Check size={16} />
              Save gate line
            </button>
            <div className="zone-list">
              {line && shown ? (
                <div>
                  <div>
                    <strong>{line.name || "gate"}</strong>
                    <small>
                      Entering {enterDirectionLabel(shown.start, shown.end, Boolean(line.flip))}
                      {" · "}
                      {line.normalized === false ? "pixels" : "fractions of the frame"}
                    </small>
                  </div>
                  <button
                    className="icon-button"
                    aria-label={`Remove ${line.name || "gate"} line`}
                    title="Remove line"
                    disabled={busy}
                    onClick={() => void removeLine()}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ) : (
                <Notice>
                  No gate line yet. Vehicle entering and exiting alerts only
                  fire once a line is drawn.
                </Notice>
              )}
            </div>
          </>
        ) : (
          <>
        <h3>Zones of interest</h3>
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
        <button
          type="button"
          className="button"
          disabled={busy}
          title="Loitering and intrusion only work inside a zone. This watches the camera's whole view."
          onClick={() => void watchWholeView()}
        >
          Watch the whole view
        </button>
        <div className="zone-list">
          {zones.map((z) => (
            <div key={z.name}>
              <div>
                <strong>{z.name}</strong>
                <small>
                  Loitering after {z.dwell_alert_seconds}s · {z.polygon.length}{" "}
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
          </>
        )}
      </aside>
    </div>
  );
}

/** The tripwire as drawn: the line, its name, and an arrow across its
 * middle pointing to the ENTER side (the same side the engine fires
 * vehicle_entry on — see lib/gate-line.ts). */
function GateLine({
  start,
  end,
  flip,
  label,
  className,
  reach,
}: {
  start: Point;
  end: Point;
  flip: boolean;
  label: string;
  className: string;
  reach: number;
}) {
  const arrow = enterArrow(start, end, flip, reach);
  const head = 8;
  const ux = (arrow.to[0] - arrow.from[0]) / (reach * 2 || 1);
  const uy = (arrow.to[1] - arrow.from[1]) / (reach * 2 || 1);
  const tip = arrow.to;
  const left: Point = [
    tip[0] - ux * head * 2 + uy * head,
    tip[1] - uy * head * 2 - ux * head,
  ];
  const right: Point = [
    tip[0] - ux * head * 2 - uy * head,
    tip[1] - uy * head * 2 + ux * head,
  ];
  return (
    <g className={className}>
      <line x1={start[0]} y1={start[1]} x2={end[0]} y2={end[1]} />
      <line
        className="enter-arrow"
        x1={arrow.from[0]}
        y1={arrow.from[1]}
        x2={tip[0]}
        y2={tip[1]}
      />
      <polygon
        className="enter-arrow-head"
        points={`${tip.join(",")} ${left.join(",")} ${right.join(",")}`}
      />
      <text x={start[0] + 6} y={start[1] - 6}>
        {label}
      </text>
    </g>
  );
}
