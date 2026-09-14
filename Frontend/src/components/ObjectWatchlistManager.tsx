import { useEffect, useMemo, useState } from "react";
import { Box, Check, RefreshCw, Upload } from "lucide-react";
import type {
  Auth,
  Camera,
  Hierarchy,
  Json,
  Mode,
  ObjectTarget,
  ObjectWatchStatus,
  Transport,
} from "../lib/types";
import { Badge, Notice, Spinner } from "./common";

const categories = ["product", "equipment", "vehicle", "ppe", "other"];

function slugify(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function dataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read image file"));
    reader.onload = () => resolve(String(reader.result || ""));
    reader.readAsDataURL(file);
  });
}

function zoneOptions(hierarchy: Hierarchy) {
  const seen = new Set<string>();
  return hierarchy.branches.flatMap((branch) =>
    branch.areas.flatMap((area) => {
      const options = [
        [area.id, area.name],
        [area.name, area.name],
      ] as const;
      return options
        .filter(([id]) => Boolean(id) && !seen.has(id))
        .map(([id, label]) => {
          seen.add(id);
          return { id, label };
        });
    }),
  );
}

function statusTone(target: ObjectTarget) {
  if (target.degraded_unavailable || target.review_state === "degraded")
    return "red";
  if (target.needs_reembed || target.review_state !== "active") return "amber";
  return "green";
}

export default function ObjectWatchlistManager({
  transport,
  authState,
  hierarchy,
  cameras = [],
  mode = "engine",
  initialTargets,
  onChange,
  notify,
}: {
  transport: Transport;
  authState: Auth;
  hierarchy: Hierarchy;
  cameras?: Camera[];
  mode?: Mode;
  initialTargets?: ObjectTarget[];
  onChange?: () => Promise<void>;
  notify?: (message: string) => void;
}) {
  const canMutate =
    mode === "engine" && authState.permissions.includes("configure_cameras");
  const zones = useMemo(() => zoneOptions(hierarchy), [hierarchy]);
  const [targets, setTargets] = useState<ObjectTarget[]>(initialTargets ?? []);
  const [label, setLabel] = useState("");
  const [category, setCategory] = useState("product");
  const [aliases, setAliases] = useState("");
  const [zoneId, setZoneId] = useState(zones[0]?.id || "");
  const [threshold, setThreshold] = useState(0.72);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [snapshotCamera, setSnapshotCamera] = useState(cameras[0]?.id || "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function refresh() {
    const value = await transport.invoke<ObjectWatchStatus>("object_targets");
    setTargets(value.targets || []);
  }

  useEffect(() => {
    if (initialTargets) return;
    let active = true;
    transport
      .invoke<ObjectWatchStatus>("object_targets")
      .then((value) => {
        if (active) setTargets(value.targets || []);
      })
      .catch((caught) => {
        if (active) setError((caught as Error).message);
      });
    return () => {
      active = false;
    };
  }, [transport, initialTargets]);

  useEffect(() => {
    if (!selectedTarget && targets[0]) setSelectedTarget(targets[0].id);
  }, [selectedTarget, targets]);

  async function run<T>(operation: () => Promise<T>, message: string) {
    setBusy(true);
    setError("");
    try {
      const result = await operation();
      await refresh();
      await onChange?.();
      notify?.(message);
      return result;
    } catch (caught) {
      setError((caught as Error).message);
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function createTarget() {
    const id = slugify(label);
    await run(
      () =>
        transport.invoke("create_object_target", [
          {
            id,
            label: label.trim(),
            category,
            aliases: aliases
              .split(",")
              .map((item) => item.trim())
              .filter(Boolean),
            allowed_zone_ids: zoneId ? [zoneId] : [],
            min_similarity: threshold,
          },
        ]),
      "Object target created",
    );
    setSelectedTarget(id);
    setLabel("");
    setAliases("");
  }

  async function uploadExample(file: File) {
    if (!selectedTarget) return;
    const image = await dataUrl(file);
    await run(
      () =>
        transport.invoke("add_object_example", [
          selectedTarget,
          image,
          [0, 0, 1, 1],
          "upload",
        ]),
      "Object example uploaded",
    );
  }

  async function enrollSnapshot() {
    if (!selectedTarget || !snapshotCamera) return;
    await run(async () => {
      const snapshot = await transport.invoke<Json>("camera_snapshot", [
        snapshotCamera,
      ]);
      const uri = String(snapshot.uri || snapshot.image_b64 || "");
      const bbox = [
        0,
        0,
        Number(snapshot.w || snapshot.width || 1),
        Number(snapshot.h || snapshot.height || 1),
      ];
      return transport.invoke("add_object_example", [
        selectedTarget,
        uri,
        bbox,
        "camera_frame",
      ]);
    }, "Camera frame enrolled");
  }

  return (
    <section className="settings-section object-watchlist">
      <div>
        <h2>Object watchlists</h2>
        <p>
          Enroll reviewed Chi products and site objects for local matching,
          tracking and rule alerts.
        </p>
      </div>
      <div>
        {mode === "demo" && (
          <Notice>
            Demo targets are fixture-backed. No local inference runs in demo
            mode.
          </Notice>
        )}
        {!canMutate && <Badge tone="amber">Read-only</Badge>}
        {error && <Notice error>{error}</Notice>}
        <div className="object-target-grid">
          {targets.length ? (
            targets.map((target) => (
              <article className="object-target-card" key={target.id}>
                <div>
                  <strong>{target.label}</strong>
                  <small>{target.id}</small>
                </div>
                <Badge tone={statusTone(target)}>
                  {target.degraded_unavailable
                    ? "Unavailable"
                    : target.needs_reembed
                      ? "Needs re-embed"
                      : target.review_state}
                </Badge>
                <p>
                  {target.category} · {target.examples.length} examples
                  {target.allowed_zone_ids.length
                    ? ` · zones ${target.allowed_zone_ids.join(", ")}`
                    : ""}
                </p>
                {canMutate && target.review_state !== "active" && (
                  <button
                    className="button small"
                    disabled={busy || target.examples.length === 0}
                    onClick={() =>
                      void run(
                        () =>
                          transport.invoke("activate_object_target", [
                            target.id,
                          ]),
                        "Object target activated",
                      )
                    }
                  >
                    <Check size={14} />
                    Activate
                  </button>
                )}
              </article>
            ))
          ) : (
            <div className="empty-inline">
              <Box size={22} />
              <span>No enrolled object targets yet.</span>
            </div>
          )}
        </div>
        {canMutate && (
          <>
            <form
              className="object-enroll-form"
              onSubmit={(event) => {
                event.preventDefault();
                void createTarget();
              }}
            >
              <label>
                Object label
                <input
                  required
                  value={label}
                  onChange={(event) => setLabel(event.target.value)}
                  placeholder="Chi carton"
                />
              </label>
              <label>
                Category
                <select
                  value={category}
                  onChange={(event) => setCategory(event.target.value)}
                >
                  {categories.map((item) => (
                    <option key={item} value={item}>
                      {item.replaceAll("_", " ")}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Allowed zone
                <select
                  value={zoneId}
                  onChange={(event) => setZoneId(event.target.value)}
                >
                  <option value="">Any reviewed zone</option>
                  {zones.map((zone) => (
                    <option key={zone.id} value={zone.id}>
                      {zone.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Aliases
                <input
                  value={aliases}
                  onChange={(event) => setAliases(event.target.value)}
                  placeholder="milk carton, loading crate"
                />
              </label>
              <label>
                Match threshold
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  value={threshold}
                  onChange={(event) => setThreshold(Number(event.target.value))}
                />
              </label>
              <button
                className="button primary"
                disabled={busy || !label.trim() || !category}
              >
                {busy ? <Spinner /> : <Box size={16} />}
                Create target
              </button>
            </form>
            <div className="object-example-actions">
              <label>
                Target for examples
                <select
                  value={selectedTarget}
                  onChange={(event) => setSelectedTarget(event.target.value)}
                >
                  {targets.map((target) => (
                    <option key={target.id} value={target.id}>
                      {target.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="file-button">
                <Upload size={16} />
                Upload example
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  disabled={busy || !selectedTarget}
                  onChange={(event) => {
                    const file = event.currentTarget.files?.[0];
                    if (file) void uploadExample(file);
                    event.currentTarget.value = "";
                  }}
                />
              </label>
              {cameras.length > 0 && (
                <>
                  <label>
                    Camera frame
                    <select
                      value={snapshotCamera}
                      onChange={(event) => setSnapshotCamera(event.target.value)}
                    >
                      {cameras.map((camera) => (
                        <option key={camera.id} value={camera.id}>
                          {camera.id}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    className="button"
                    disabled={busy || !selectedTarget || !snapshotCamera}
                    onClick={() => void enrollSnapshot()}
                  >
                    Enroll object from frame
                  </button>
                </>
              )}
              <button
                className="button"
                disabled={busy}
                onClick={() =>
                  void run(
                    () => transport.invoke("reembed_object_targets", ["hash"]),
                    "Object embeddings refreshed",
                  )
                }
              >
                <RefreshCw size={16} />
                Re-embed locally
              </button>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
