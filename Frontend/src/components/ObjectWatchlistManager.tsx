import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Box,
  Check,
  Crop,
  Eye,
  Power,
  PowerOff,
  RefreshCw,
  Upload,
} from "lucide-react";
import type {
  Auth,
  Camera,
  Hierarchy,
  Json,
  Mode,
  ObjectExample,
  ObjectWatchJobStatus,
  ObjectTarget,
  ObjectWatchRuntime,
  ObjectWatchStatus,
  Transport,
  Zone,
} from "../lib/types";
import { Badge, Notice, Spinner } from "./common";

export const objectCategories = [
  "product",
  "vehicle",
  "pallet",
  "ppe",
  "custom",
] as const;
type BBox = [number, number, number, number];
type DraftImage = {
  uri: string;
  source: string;
  name: string;
  width: number;
  height: number;
};

const reasonText: Record<string, string> = {
  missing_local_model: "Choose the local semantic model files below.",
  model_unavailable: "The configured local semantic model is not available.",
  no_reviewed_positive_examples: "Review at least one positive example.",
  no_embeddings: "Run re-embed after reviewing examples.",
  target_not_active: "Activate this target when it is ready.",
};

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

function imageSize(uri: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () =>
      resolve({
        width: image.naturalWidth || 1,
        height: image.naturalHeight || 1,
      });
    image.onerror = () => reject(new Error("Could not read image dimensions"));
    image.src = uri;
  });
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, Number.isFinite(value) ? value : min));
}

function normalizeBox(box: BBox, width: number, height: number): BBox {
  const x1 = clamp(Math.min(box[0], box[2]), 0, width);
  const y1 = clamp(Math.min(box[1], box[3]), 0, height);
  const x2 = clamp(Math.max(box[0], box[2]), 0, width);
  const y2 = clamp(Math.max(box[1], box[3]), 0, height);
  return [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)];
}

function isSupportedImage(file: File) {
  const type = file.type.toLowerCase();
  const name = file.name.toLowerCase();
  return (
    type === "image/png" ||
    type === "image/jpeg" ||
    name.endsWith(".png") ||
    name.endsWith(".jpg") ||
    name.endsWith(".jpeg")
  );
}

function statusTone(target: ObjectTarget) {
  if (target.degraded_unavailable || target.review_state === "degraded")
    return "red";
  if (target.needs_reembed || target.review_state !== "active") return "amber";
  return "green";
}

function runtimeCopy(runtime?: ObjectWatchRuntime) {
  if (!runtime) return "Runtime status has not been loaded yet.";
  const backend = runtime.backend || "semantic";
  if (runtime.executable_verified === true)
    return `${backend} verified for matching${runtime.fingerprint ? ` · ${runtime.fingerprint}` : ""}`;
  if (
    runtime.status === "structurally_available" ||
    runtime.structurally_available ||
    runtime.status === "ready"
  )
    return `${backend} configured — prepare examples to verify execution${runtime.fingerprint ? ` · ${runtime.fingerprint}` : ""}`;
  if (runtime.status === "degraded")
    return `${backend} configured with warnings. Prepare examples after reviewing the warnings below.`;
  const reasons = runtime.reason_codes?.length
    ? runtime.reason_codes.map((code) => reasonText[code] || code).join(" ")
    : "Set the local semantic model path. Hash fallback is not used for enrollment.";
  return `${backend} unavailable. ${reasons}`;
}

function canPrepareRuntime(runtime?: ObjectWatchRuntime) {
  if (!runtime) return false;
  if (runtime.status === "demo" || runtime.status === "unavailable")
    return false;
  return (
    runtime.status === "ready" ||
    runtime.status === "structurally_available" ||
    runtime.status === "degraded" ||
    runtime.structurally_available === true
  );
}

function prepareDisabledReason(
  runtime: ObjectWatchRuntime | undefined,
  busy: boolean,
  pending: boolean,
) {
  if (busy) return "Another object watchlist action is still running.";
  if (pending) return "Preparation is already queued or running.";
  if (!runtime) return "Runtime status is still loading.";
  if (canPrepareRuntime(runtime)) return "";
  const reasons = runtime.reason_codes?.length
    ? runtime.reason_codes.map((code) => reasonText[code] || code).join(" ")
    : "Configure local semantic model paths in Advanced model setup.";
  return reasons;
}

function exampleRows(target: ObjectTarget) {
  return [
    ...target.examples.map((example) => ({ example, negative: false })),
    ...target.negative_examples.map((example) => ({ example, negative: true })),
  ];
}

export default function ObjectWatchlistManager({
  transport,
  authState,
  hierarchy,
  cameras = [],
  mode = "engine",
  initialTargets,
  initialRuntime,
  onChange,
  notify,
}: {
  transport: Transport;
  authState: Auth;
  hierarchy: Hierarchy;
  cameras?: Camera[];
  mode?: Mode;
  initialTargets?: ObjectTarget[];
  initialRuntime?: ObjectWatchRuntime;
  onChange?: () => Promise<void>;
  notify?: (message: string) => void;
}) {
  const canEnroll =
    mode === "engine" && authState.permissions.includes("configure_cameras");
  const canEditRules =
    mode === "engine" && authState.permissions.includes("configure_detectors");
  const canMutate = canEnroll || canEditRules;
  const [runtime, setRuntime] = useState<ObjectWatchRuntime | undefined>(
    initialRuntime,
  );
  const [targets, setTargets] = useState<ObjectTarget[]>(initialTargets ?? []);
  const [label, setLabel] = useState("");
  const [category, setCategory] = useState("product");
  const [aliases, setAliases] = useState("");
  const [grounding, setGrounding] = useState("");
  const [threshold, setThreshold] = useState(0.72);
  const [selectedTarget, setSelectedTarget] = useState(
    initialTargets?.[0]?.id || "",
  );
  const [snapshotCamera, setSnapshotCamera] = useState(cameras[0]?.id || "");
  const [ruleCamera, setRuleCamera] = useState(cameras[0]?.id || "");
  const [ruleZone, setRuleZone] = useState("");
  const [cameraZones, setCameraZones] = useState<Zone[]>([]);
  const [modelPath, setModelPath] = useState("");
  const [worldWeights, setWorldWeights] = useState("");
  const [clipWeights, setClipWeights] = useState("");
  const [draft, setDraft] = useState<DraftImage | null>(null);
  const [bbox, setBbox] = useState<BBox>([0, 0, 1, 1]);
  const [negative, setNegative] = useState(false);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [reembedJob, setReembedJob] = useState<ObjectWatchJobStatus | null>(
    null,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const imageRef = useRef<HTMLImageElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragStart = useRef<{ x: number; y: number } | null>(null);

  const selected = targets.find((target) => target.id === selectedTarget);
  const runtimeVerified =
    runtime?.executable_verified === true ||
    (runtime?.status === "ready" && runtime?.executable_verified !== false);
  const canPrepare = canPrepareRuntime(runtime);
  const reembedPending = ["queued", "running"].includes(
    reembedJob?.status || "",
  );
  const prepareReason = prepareDisabledReason(runtime, busy, reembedPending);
  const selectedCanActivate = Boolean(
    selected?.can_activate ?? selected?.ready_for_activation,
  );
  const currentCamera = cameras.find((camera) => camera.id === ruleCamera);
  const enabledRule = Boolean(
    currentCamera?.object_watch_rules?.some(
      (rule: Json) =>
        rule?.trigger?.object_id === selectedTarget &&
        (ruleZone ? rule?.trigger?.zone === ruleZone : !rule?.trigger?.zone),
    ),
  );

  async function refresh() {
    const value = await transport.invoke<ObjectWatchStatus>("object_targets");
    setRuntime(value.runtime);
    setTargets(value.targets || []);
  }

  useEffect(() => {
    setRuntime(initialRuntime);
    if (initialTargets) setTargets(initialTargets);
  }, [initialRuntime, initialTargets]);

  useEffect(() => {
    if (initialTargets) return;
    let active = true;
    transport
      .invoke<ObjectWatchStatus>("object_targets")
      .then((value) => {
        if (!active) return;
        setRuntime(value.runtime);
        setTargets(value.targets || []);
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

  useEffect(() => {
    if (!snapshotCamera && cameras[0]) setSnapshotCamera(cameras[0].id);
    if (!ruleCamera && cameras[0]) setRuleCamera(cameras[0].id);
  }, [cameras, ruleCamera, snapshotCamera]);

  useEffect(() => {
    if (!ruleCamera || mode !== "engine") return;
    let active = true;
    transport
      .invoke<Zone[]>("list_zones", [ruleCamera])
      .then((value) => {
        if (active) setCameraZones(Array.isArray(value) ? value : []);
      })
      .catch(() => {
        if (active) setCameraZones([]);
      });
    return () => {
      active = false;
    };
  }, [mode, ruleCamera, transport]);

  useEffect(() => {
    if (!reembedPending || !reembedJob?.job_id) return;
    let active = true;
    const timer = setTimeout(() => {
      transport
        .invoke<ObjectWatchJobStatus>("object_watch_job_status", [
          reembedJob.job_id,
        ])
        .then(async (job) => {
          if (!active) return;
          setReembedJob(job);
          if (job.status === "completed") {
            await refresh();
            await onChange?.();
            notify?.("Object embeddings refreshed");
          } else if (job.status === "failed") {
            setError(
              job.error || job.message || "Object embedding job failed.",
            );
          }
        })
        .catch((caught) => {
          if (active) setError((caught as Error).message);
        });
    }, 1500);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [onChange, notify, reembedJob, reembedPending, transport]);

  async function run<T>(operation: () => Promise<T>, message?: string) {
    setBusy(true);
    setError("");
    try {
      const result = await operation();
      await refresh();
      await onChange?.();
      if (message) notify?.(message);
      return result;
    } catch (caught) {
      setError((caught as Error).message);
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function createTarget() {
    const object_id = slugify(label);
    if (!object_id) return setError("Add a short object label first.");
    await run(
      () =>
        transport.invoke("create_object_target", [
          object_id,
          label.trim(),
          category,
          aliases
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean),
          threshold,
          [],
          grounding.trim(),
        ]),
      "Object target created",
    );
    setSelectedTarget(object_id);
    setLabel("");
    setAliases("");
    setGrounding("");
  }

  async function setUploadedFile(file?: File | null) {
    setError("");
    if (!file) {
      setError(
        "No image selected. Choose a JPEG or PNG photo when you are ready.",
      );
      return;
    }
    if (!isSupportedImage(file)) {
      setError(
        "Use a JPEG or PNG image. HEIC and other formats are not supported yet.",
      );
      return;
    }
    try {
      const uri = await dataUrl(file);
      const size = await imageSize(uri);
      setDraft({ uri, source: "upload", name: file.name, ...size });
      setBbox([0, 0, size.width, size.height]);
    } catch (caught) {
      setError((caught as Error).message);
    }
  }

  async function loadSnapshot() {
    if (!snapshotCamera) return;
    await run(async () => {
      const snapshot = await transport.invoke<Json>("camera_snapshot", [
        snapshotCamera,
      ]);
      if (snapshot.error) throw new Error(String(snapshot.error));
      const uri = String(snapshot.uri || snapshot.image_b64 || "");
      if (!uri) throw new Error("Camera snapshot did not include an image.");
      const width = Number(snapshot.w || snapshot.width || 1);
      const height = Number(snapshot.h || snapshot.height || 1);
      setDraft({
        uri,
        source: `camera:${snapshotCamera}`,
        name: snapshotCamera,
        width,
        height,
      });
      setBbox([0, 0, width, height]);
      return snapshot;
    });
  }

  async function saveExample() {
    if (!draft) {
      setError("Choose a photo before saving an example.");
      return;
    }
    let targetId = selectedTarget;
    if (!targetId) {
      targetId = slugify(label);
      if (!targetId) {
        setError("Name the object before saving this photo.");
        return;
      }
      const exists = targets.some((target) => target.id === targetId);
      if (!exists) {
        const created = await run<Json>(
          () =>
            transport.invoke("create_object_target", [
              targetId,
              label.trim(),
              category,
              aliases
                .split(",")
                .map((item) => item.trim())
                .filter(Boolean),
              threshold,
              [],
              grounding.trim(),
            ]),
          "Object target created",
        );
        if (!created) return;
      }
      setSelectedTarget(targetId);
    }
    const clean = normalizeBox(bbox, draft.width, draft.height);
    if (clean[2] <= clean[0] || clean[3] <= clean[1]) {
      setError(
        "Crop needs width and height. Use full image if the whole object is visible.",
      );
      return;
    }
    const result = await run<Json>(
      () =>
        transport.invoke("add_object_example", [
          targetId,
          draft.uri,
          clean,
          draft.source,
          "pixel_xyxy",
          negative,
        ]),
      negative ? "Negative example saved" : "Positive example saved",
    );
    if (!result) return;
    const example = result?.example as ObjectExample | undefined;
    if (example?.id) await loadPreview(targetId, example.id);
    setDraft(null);
    setNegative(false);
    setLabel("");
    setAliases("");
    setGrounding("");
  }

  async function startReembed() {
    setBusy(true);
    setError("");
    try {
      const result = await transport.invoke<Json>("reembed_object_targets");
      const jobId = String(result?.job_id || "");
      const status = String(result?.status || (jobId ? "queued" : "completed"));
      if (jobId) {
        setReembedJob({ job_id: jobId, status });
        if (status === "completed") {
          await refresh();
          await onChange?.();
          notify?.("Object embeddings refreshed");
        } else if (status === "failed") {
          setError(
            String(
              result.error || result.message || "Object embedding job failed.",
            ),
          );
        }
      } else {
        setReembedJob({ job_id: "", status: "completed" });
        await refresh();
        await onChange?.();
        notify?.("Object embeddings refreshed");
      }
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function loadPreview(objectId: string, exampleId: string) {
    const value = await run<Json>(() =>
      transport.invoke("object_example_preview", [objectId, exampleId]),
    );
    if (value?.image_b64) {
      setPreviews((current) => ({
        ...current,
        [exampleId]: `data:${value.mime_type || "image/png"};base64,${value.image_b64}`,
      }));
    }
  }

  function imagePoint(
    event: React.PointerEvent,
  ): { x: number; y: number } | null {
    if (!draft || !imageRef.current) return null;
    const rect = imageRef.current.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * draft.width;
    const y = ((event.clientY - rect.top) / rect.height) * draft.height;
    return { x: clamp(x, 0, draft.width), y: clamp(y, 0, draft.height) };
  }

  const cropStyle = draft
    ? {
        left: `${(Math.min(bbox[0], bbox[2]) / draft.width) * 100}%`,
        top: `${(Math.min(bbox[1], bbox[3]) / draft.height) * 100}%`,
        width: `${(Math.abs(bbox[2] - bbox[0]) / draft.width) * 100}%`,
        height: `${(Math.abs(bbox[3] - bbox[1]) / draft.height) * 100}%`,
      }
    : undefined;

  return (
    <section className="settings-section object-watchlist">
      <div>
        <h2>Object watchlists</h2>
        <p>
          Enroll examples, approve positives and negatives, then separately turn
          on object_seen alerts for each camera and zone.
        </p>
      </div>
      <div>
        {mode === "demo" && (
          <Notice>
            Demo targets are fixture-backed. No local inference runs in demo
            mode.
          </Notice>
        )}
        <div className="object-runtime-card">
          <div>
            <strong>Semantic model</strong>
            <small>{runtimeCopy(runtime)}</small>
          </div>
          <Badge
            tone={runtimeVerified ? "green" : canPrepare ? "amber" : "amber"}
          >
            {runtimeVerified
              ? "Verified"
              : canPrepare
                ? "Configured"
                : "Unavailable"}
          </Badge>
        </div>
        {!canMutate && (
          <>
            <Badge tone="amber">Read-only</Badge>
            <Notice>
              You can review configured object targets, but only owners or
              installers can upload examples, prepare embeddings, or enable
              camera alerts.
            </Notice>
          </>
        )}
        {error && <Notice error>{error}</Notice>}
        <div className="object-target-grid">
          {targets.length ? (
            targets.map((target) => {
              const approvedPositives = target.examples.filter(
                (e) => e.reviewed,
              ).length;
              const approvedNegatives = target.negative_examples.filter(
                (e) => e.reviewed,
              ).length;
              return (
                <article className="object-target-card" key={target.id}>
                  <div className="object-target-head">
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
                  </div>
                  <p>
                    {target.category} · {approvedPositives} approved positive
                    {approvedNegatives
                      ? ` · ${approvedNegatives} approved negative`
                      : ""}
                    {target.allowed_zone_ids.length
                      ? ` · target zones ${target.allowed_zone_ids.join(", ")}`
                      : ""}
                  </p>
                  {target.grounding_description && (
                    <small>{target.grounding_description}</small>
                  )}
                  {target.reasons?.length ? (
                    <ul className="object-reasons">
                      {target.reasons.map((reason) => (
                        <li key={reason}>{reasonText[reason] || reason}</li>
                      ))}
                    </ul>
                  ) : null}
                  <div className="object-card-actions">
                    <button
                      className="button small"
                      onClick={() => setSelectedTarget(target.id)}
                    >
                      <Eye size={14} />
                      Review
                    </button>
                    {canEnroll && target.review_state !== "active" && (
                      <button
                        className="button small"
                        disabled={
                          busy ||
                          !Boolean(
                            target.can_activate ?? target.ready_for_activation,
                          )
                        }
                        title={
                          Boolean(
                            target.can_activate ?? target.ready_for_activation,
                          )
                            ? "Recognize this object"
                            : "Resolve readiness reasons before activation"
                        }
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
                        Recognize this object
                      </button>
                    )}
                    {canEnroll && target.review_state === "active" && (
                      <button
                        className="button small"
                        disabled={busy}
                        onClick={() =>
                          void run(
                            () =>
                              transport.invoke("deactivate_object_target", [
                                target.id,
                              ]),
                            "Object target deactivated",
                          )
                        }
                      >
                        <PowerOff size={14} />
                        Deactivate
                      </button>
                    )}
                  </div>
                </article>
              );
            })
          ) : (
            <div className="empty-inline">
              <Box size={22} />
              <span>No enrolled object targets yet.</span>
            </div>
          )}
        </div>
        {canMutate && (
          <>
            {canEnroll && (
              <div className="object-enrollment-flow">
                <div
                  className="object-step-strip"
                  aria-label="Enrollment steps"
                >
                  <span className="object-step-pill active">
                    1 Photo + name
                  </span>
                  <span className="object-step-pill">2 Crop + save</span>
                  <span className="object-step-pill">
                    3 Prepare recognition
                  </span>
                  <span className="object-step-pill">4 Camera alert</span>
                </div>
                <div className="object-step-card object-step-card--hero">
                  <div className="object-step-kicker">Step 1</div>
                  <h3>Add the object photo</h3>
                  <p>
                    Choose a clear JPEG or PNG, then save it to an existing
                    object or name a new one. Cropping comes next.
                  </p>
                  <div className="object-enroll-form">
                    <input
                      ref={fileInputRef}
                      className="visually-hidden-file"
                      type="file"
                      tabIndex={-1}
                      aria-hidden="true"
                      accept="image/png,image/jpeg"
                      onChange={(event) => {
                        void setUploadedFile(event.currentTarget.files?.[0]);
                        event.currentTarget.value = "";
                      }}
                    />
                    <button
                      className="button primary object-photo-button"
                      type="button"
                      disabled={busy}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload size={16} />
                      Choose photo
                    </button>
                    <small className="object-photo-note">
                      {draft
                        ? `${draft.name} loaded for cropping`
                        : "No photo chosen yet"}
                    </small>
                    {targets.length > 0 && (
                      <label>
                        Save to
                        <select
                          value={selectedTarget}
                          onChange={(event) =>
                            setSelectedTarget(event.target.value)
                          }
                        >
                          <option value="">New object</option>
                          {targets.map((target) => (
                            <option key={target.id} value={target.id}>
                              {target.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    {!selectedTarget && (
                      <>
                        <label>
                          Object name
                          <input
                            value={label}
                            onChange={(event) => setLabel(event.target.value)}
                            placeholder="Yellow cup"
                          />
                        </label>
                        <label>
                          Category
                          <select
                            value={category}
                            onChange={(event) =>
                              setCategory(event.target.value)
                            }
                          >
                            {objectCategories.map((item) => (
                              <option key={item} value={item}>
                                {item.replaceAll("_", " ")}
                              </option>
                            ))}
                          </select>
                        </label>
                      </>
                    )}
                    {cameras.length > 0 && (
                      <>
                        <label>
                          Or use camera
                          <select
                            value={snapshotCamera}
                            onChange={(event) =>
                              setSnapshotCamera(event.target.value)
                            }
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
                          type="button"
                          disabled={busy || !snapshotCamera}
                          onClick={() => void loadSnapshot()}
                        >
                          <Crop size={16} />
                          Use snapshot
                        </button>
                      </>
                    )}
                  </div>
                  {!selectedTarget && (
                    <details className="object-advanced-details">
                      <summary>Optional naming details</summary>
                      <div className="object-enroll-form">
                        <label>
                          Aliases
                          <input
                            value={aliases}
                            onChange={(event) => setAliases(event.target.value)}
                            placeholder="cup, yellow mug"
                          />
                        </label>
                        <label>
                          Visual description
                          <textarea
                            value={grounding}
                            onChange={(event) =>
                              setGrounding(event.target.value)
                            }
                            placeholder="Plain visual description, not a private product name."
                          />
                        </label>
                        <label>
                          Match strictness
                          <input
                            type="number"
                            min="0"
                            max="1"
                            step="0.01"
                            value={threshold}
                            onChange={(event) =>
                              setThreshold(Number(event.target.value))
                            }
                          />
                        </label>
                      </div>
                    </details>
                  )}
                </div>
              </div>
            )}
            {draft && (
              <div className="object-crop-panel">
                <div>
                  <h3>Crop and save</h3>
                  <p>
                    Drag around the object. Use full image if the photo already
                    frames it cleanly. The saved preview comes from the backend.
                  </p>
                </div>
                <div
                  className="crop-stage"
                  onPointerDown={(event) => {
                    const point = imagePoint(event);
                    if (!point) return;
                    dragStart.current = point;
                    setBbox([point.x, point.y, point.x, point.y]);
                    event.currentTarget.setPointerCapture(event.pointerId);
                  }}
                  onPointerMove={(event) => {
                    if (!dragStart.current) return;
                    const point = imagePoint(event);
                    if (point)
                      setBbox([
                        dragStart.current.x,
                        dragStart.current.y,
                        point.x,
                        point.y,
                      ]);
                  }}
                  onPointerUp={() => {
                    dragStart.current = null;
                    setBbox((current) =>
                      normalizeBox(current, draft.width, draft.height),
                    );
                  }}
                >
                  <img
                    ref={imageRef}
                    src={draft.uri}
                    alt={`Crop ${draft.name}`}
                  />
                  <span className="crop-box" style={cropStyle} />
                </div>
                <div className="crop-fields">
                  <label className="checkbox-line">
                    <input
                      type="checkbox"
                      checked={negative}
                      onChange={(event) => setNegative(event.target.checked)}
                    />
                    Negative example
                  </label>
                  <button
                    className="button"
                    type="button"
                    onClick={() => setBbox([0, 0, draft.width, draft.height])}
                  >
                    Use full image
                  </button>
                  <button
                    className="button primary"
                    disabled={
                      busy || !draft || (!selectedTarget && !label.trim())
                    }
                    onClick={() => void saveExample()}
                  >
                    Save photo and example
                  </button>
                  <details className="object-advanced-details crop-coordinate-details">
                    <summary>Advanced crop coordinates</summary>
                    <div className="crop-coordinate-grid">
                      {(["x1", "y1", "x2", "y2"] as const).map(
                        (name, index) => (
                          <label key={name}>
                            {name.toUpperCase()}
                            <input
                              type="number"
                              value={Math.round(bbox[index])}
                              min={0}
                              max={index % 2 === 0 ? draft.width : draft.height}
                              onChange={(event) => {
                                const next = [...bbox] as BBox;
                                next[index] = Number(event.target.value);
                                setBbox(next);
                              }}
                            />
                          </label>
                        ),
                      )}
                    </div>
                  </details>
                </div>
              </div>
            )}
            {selected && (
              <div className="object-review-panel">
                <div>
                  <h3>Review examples for {selected.label}</h3>
                  <p>
                    Mark true examples and look-alikes before preparing
                    recognition. Camera alerts are enabled separately.
                  </p>
                </div>
                <div className="object-example-list">
                  {exampleRows(selected).length ? (
                    exampleRows(selected).map(({ example, negative }) => (
                      <article className="object-example-row" key={example.id}>
                        {previews[example.id] ? (
                          <img
                            src={previews[example.id]}
                            alt="Saved crop preview"
                          />
                        ) : (
                          <div className="crop-placeholder">
                            <Crop size={18} />
                          </div>
                        )}
                        <div>
                          <strong>{negative ? "Negative" : "Positive"}</strong>
                          <small>
                            {example.source} · {example.bbox_format || "legacy"}{" "}
                            · [{example.bbox.join(", ")}]
                          </small>
                          <Badge tone={example.reviewed ? "green" : "amber"}>
                            {example.reviewed ? "Reviewed" : "Unreviewed"}
                          </Badge>
                        </div>
                        <button
                          className="button small"
                          disabled={busy}
                          onClick={() =>
                            void loadPreview(selected.id, example.id)
                          }
                        >
                          Preview saved crop
                        </button>
                        {canEnroll && (
                          <button
                            className="button small"
                            disabled={busy}
                            onClick={() =>
                              void run(
                                () =>
                                  transport.invoke("review_object_example", [
                                    selected.id,
                                    example.id,
                                    !example.reviewed,
                                  ]),
                                example.reviewed
                                  ? "Example marked unreviewed"
                                  : "Example reviewed",
                              )
                            }
                          >
                            {example.reviewed
                              ? "Mark unreviewed"
                              : "Mark reviewed"}
                          </button>
                        )}
                      </article>
                    ))
                  ) : (
                    <div className="empty-inline">
                      <AlertTriangle size={20} />
                      <span>Add positive and negative cropped examples.</span>
                    </div>
                  )}
                </div>
              </div>
            )}
            <div className="object-runtime-actions">
              {canEnroll && (
                <>
                  <details className="object-advanced-details object-model-details">
                    <summary>Advanced model setup</summary>
                    <div className="object-enroll-form">
                      <label>
                        Model path
                        <input
                          value={modelPath}
                          onChange={(event) => setModelPath(event.target.value)}
                          placeholder="/models/siglip"
                        />
                      </label>
                      <label>
                        World weights
                        <input
                          value={worldWeights}
                          onChange={(event) =>
                            setWorldWeights(event.target.value)
                          }
                          placeholder="optional local path"
                        />
                      </label>
                      <label>
                        CLIP weights
                        <input
                          value={clipWeights}
                          onChange={(event) =>
                            setClipWeights(event.target.value)
                          }
                          placeholder="optional local path"
                        />
                      </label>
                      <button
                        className="button"
                        disabled={
                          busy || (!modelPath && !worldWeights && !clipWeights)
                        }
                        onClick={() =>
                          void run(
                            () =>
                              transport.invoke(
                                "set_object_watch_runtime_config",
                                [
                                  {
                                    backend: "siglip",
                                    model_path: modelPath || undefined,
                                    world_weights: worldWeights || undefined,
                                    clip_weights: clipWeights || undefined,
                                  },
                                ],
                              ),
                            "Object watch runtime saved",
                          )
                        }
                      >
                        Save local model paths
                      </button>
                    </div>
                  </details>
                  <button
                    className="button"
                    disabled={busy || !canPrepare || reembedPending}
                    title={prepareReason || "Prepare reviewed examples"}
                    onClick={() => void startReembed()}
                  >
                    {reembedPending ? <Spinner /> : <RefreshCw size={16} />}
                    {reembedPending
                      ? `Preparing ${reembedJob?.status}`
                      : "Prepare recognition"}
                  </button>
                  {prepareReason && (
                    <small className="object-action-hint">
                      {prepareReason}
                    </small>
                  )}
                </>
              )}
              {canEditRules && selected && (
                <>
                  <label>
                    Camera for alert rule
                    <select
                      value={ruleCamera}
                      onChange={(event) => {
                        setRuleCamera(event.target.value);
                        setRuleZone("");
                      }}
                    >
                      <option value="">Choose a camera</option>
                      {cameras.map((camera) => (
                        <option key={camera.id} value={camera.id}>
                          {camera.id}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Zone for alert rule
                    <select
                      value={ruleZone}
                      onChange={(event) => setRuleZone(event.target.value)}
                      disabled={!ruleCamera}
                    >
                      <option value="">Whole camera view</option>
                      {cameraZones.map((zone) => (
                        <option key={zone.name} value={zone.name}>
                          {zone.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button
                    className="button primary"
                    disabled={busy || !selectedTarget || !ruleCamera}
                    onClick={() =>
                      void run(
                        () =>
                          transport.invoke("set_object_watch_rule", [
                            ruleCamera,
                            selectedTarget,
                            true,
                            ruleZone || undefined,
                          ]),
                        "object_seen alert rule enabled",
                      )
                    }
                  >
                    <Power size={16} />
                    Enable object_seen alert
                  </button>
                  <button
                    className="button"
                    disabled={
                      busy || !selectedTarget || !ruleCamera || !enabledRule
                    }
                    onClick={() =>
                      void run(
                        () =>
                          transport.invoke("set_object_watch_rule", [
                            ruleCamera,
                            selectedTarget,
                            false,
                            ruleZone || undefined,
                          ]),
                        "object_seen alert rule disabled",
                      )
                    }
                  >
                    Disable this alert rule
                  </button>
                  {selected &&
                    !selectedCanActivate &&
                    selected.review_state !== "active" && (
                      <small className="object-action-hint object-rule-warning">
                        Camera alert setup is separate. This target is still
                        draft or unprepared, so prepare and recognize it before
                        relying on object_seen alerts.
                      </small>
                    )}
                </>
              )}
            </div>
          </>
        )}
      </div>
    </section>
  );
}
