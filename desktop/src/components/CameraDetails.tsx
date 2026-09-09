import { useEffect, useState } from "react";
import {
  Check,
  RotateCw,
  Save,
  Plus,
  Trash2,
  ShieldCheck,
  Pause,
  ChevronRight,
} from "lucide-react";
import {
  DETECTORS,
  type Camera,
  type Json,
  type Scene,
  type Transport,
} from "../lib/types";
import { Badge, CameraMedia, Notice, Spinner, Toggle } from "./common";
import ZoneEditor from "./ZoneEditor";
export default function CameraDetails({
  camera,
  api,
  stream,
  initialTab = "scene",
  onChange,
  notify,
  editable = true,
  onDirtyChange,
}: {
  editable?: boolean;
  camera: Camera;
  api: Transport;
  stream?: string;
  initialTab?: string;
  onChange: () => Promise<void>;
  notify: (s: string) => void;
  onDirtyChange?: (dirty: boolean) => void;
}) {
  const [tab, setTab] = useState(initialTab);
  const [scene, setScene] = useState<Scene | null>(null);
  const [draft, setDraft] = useState<Scene | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [question, setQuestion] = useState("");
  const [dwell, setDwell] = useState(4);
  const [areas, setAreas] = useState<Json[]>([]);
  const [english, setEnglish] = useState<Json>({});
  const [dirty, setDirty] = useState(false);
  const [presets, setPresets] = useState<Json>({});
  useEffect(() => {
    onDirtyChange?.(dirty || Boolean(question.trim()));
  }, [dirty, question, onDirtyChange]);
  useEffect(() => {
    let active = true;
    setLoading(true);
    Promise.all([
      api.invoke<Scene | null>("scene_context", [camera.id]),
      api.invoke<Json[]>("list_areas"),
      api.invoke<Json>("english_rules_status"),
      api.invoke<Json>("presets"),
    ])
      .then(([s, a, e, p]) => {
        if (active) {
          setScene(s);
          setDraft(s);
          setDirty(false);
          setAreas(a);
          setEnglish(e);
          setPresets(p);
        }
      })
      .catch((e) => active && setError(e.message))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [api, camera.id]);
  async function act(method: string, args: unknown[], message: string) {
    setBusy(true);
    setError("");
    try {
      const res = await api.invoke(method, args);
      if (res?.ok === false) throw new Error(res.message || "Operation failed");
      if (res?.context) {
        setScene(res.context);
        setDraft(res.context);
        setDirty(false);
      }
      if (method === "add_custom_rule") setQuestion("");
      await onChange();
      notify(message);
      return true;
    } catch (e) {
      setError((e as Error).message);
      return false;
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    if (dirty || tab !== "scene") return;
    let active = true;
    const timer = setInterval(() => {
      api
        .invoke<Scene | null>("scene_context", [camera.id])
        .then((s) => {
          if (active) {
            setScene(s);
            setDraft(s);
          }
        })
        .catch(() => {});
    }, 4000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [api, camera.id, dirty, tab]);
  const edit = (key: keyof Scene, value: any) => {
    if (draft) {
      setDraft({ ...draft, [key]: value });
      setDirty(true);
    }
  };
  const reviewed = scene?.mapping.status === "ready_reviewed";
  const hasEvidence = Boolean(scene?.source_frame_uri);
  const custom = camera.custom_rules || [];
  const scanner = english.cameras?.[camera.id];
  return (
    <>
      <div className="detail-tabs" role="tablist">
        {[
          ["scene", "Scene context"],
          ["detectors", "Detectors"],
          ["rules", "English rules"],
          ["zones", "Zones"],
        ].map(([key, label]) => (
          <button
            key={key}
            role="tab"
            aria-selected={key === tab}
            className={key === tab ? "active" : ""}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>
      <fieldset disabled={!editable}>
        {error && <Notice error>{error}</Notice>}
        {loading ? (
          <div className="loading">
            <Spinner />
            Loading camera configuration
          </div>
        ) : (
          <>
            {tab === "scene" && (
              <>
                <div className="scene-image">
                  {scene?.source_frame_uri ? (
                    <img
                      src={scene.source_frame_uri}
                      alt="Frame used for scene mapping"
                    />
                  ) : (
                    <CameraMedia camera={camera} stream={stream} paused />
                  )}
                </div>
                <div className="context-status">
                  <Badge tone={reviewed ? "green" : "amber"}>
                    {reviewed ? "Human reviewed" : "Needs review"}
                  </Badge>
                  <span>
                    {scene?.mapping.provenance === "demo"
                      ? "Sample scene"
                      : scene?.mapping.provenance || "No mapping yet"}
                  </span>
                  {scene && (
                    <span>
                      Model confidence {Math.round(scene.confidence * 100)}%
                    </span>
                  )}
                </div>
                {!hasEvidence && (
                  <Notice>
                    Mapping evidence is missing. Remap the camera to obtain a
                    source frame before approval.
                  </Notice>
                )}
                {draft ? (
                  <>
                    <label>
                      Environment
                      <select
                        value={draft.environment_type}
                        onChange={(e) =>
                          edit("environment_type", e.target.value)
                        }
                      >
                        {Array.from(
                          new Set([
                            draft.environment_type,
                            "unknown",
                            "estate_gate",
                            "estate_street",
                            "perimeter_fence",
                            "retail_shop",
                            "mall_corridor",
                            "office_lobby",
                            "office_floor",
                            "parking_lot",
                            "banking_hall",
                            "atm_area",
                            "warehouse_floor",
                            "generator_area",
                            "residential_interior",
                            "residential_exterior",
                          ]),
                        ).map((v) => (
                          <option key={v} value={v}>
                            {v.replaceAll("_", " ")}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Expected actors
                      <input
                        value={draft.expected_actors.join(", ")}
                        onChange={(e) =>
                          edit(
                            "expected_actors",
                            e.target.value.split(",").map((s) => s.trim()),
                          )
                        }
                        placeholder="Staff, customers, vehicles"
                      />
                    </label>
                    <label>
                      Scene description
                      <textarea
                        value={draft.scene_description}
                        onChange={(e) =>
                          edit("scene_description", e.target.value)
                        }
                        rows={4}
                      />
                    </label>
                  </>
                ) : (
                  <p className="empty-copy">
                    This camera has no scene mapping yet.
                  </p>
                )}
                <label>
                  Area
                  <select
                    value={camera.area_id || ""}
                    disabled={busy}
                    onChange={(e) =>
                      void act(
                        "assign_camera_area",
                        [camera.id, e.target.value],
                        "Camera area updated",
                      )
                    }
                  >
                    <option value="">Ungrouped</option>
                    {areas.map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.name || a.id}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="actions">
                  <button
                    className="button primary"
                    disabled={busy || !draft || !hasEvidence}
                    onClick={() =>
                      void act(
                        "approve_scene_context",
                        [camera.id, draft],
                        "Scene context approved",
                      )
                    }
                  >
                    <ShieldCheck size={16} />
                    {dirty ? "Save & approve" : "Approve context"}
                  </button>
                  <button
                    className="button"
                    disabled={busy || !dirty}
                    onClick={() =>
                      void act(
                        "update_scene_context",
                        [camera.id, draft],
                        "Draft context saved",
                      )
                    }
                  >
                    <Save size={16} />
                    Save draft
                  </button>
                  <button
                    className="button"
                    disabled={busy}
                    onClick={async () => {
                      if (
                        dirty &&
                        !confirm(
                          "Discard unsaved scene edits and request remapping?",
                        )
                      )
                        return;
                      await act(
                        "enqueue_scene_mapping",
                        [[camera.id]],
                        "Scene mapping queued",
                      );
                    }}
                  >
                    <RotateCw size={16} />
                    Remap
                  </button>
                </div>
                {dirty && (
                  <p className="field-note">
                    Unsaved changes. Background refresh will not overwrite this
                    draft.
                  </p>
                )}
                <div className="suggestions">
                  {(scene?.zones || []).map((z) => (
                    <div className="suggestion" key={z.id}>
                      <div>
                        <strong>{z.id}</strong>
                        <small>Suggested {z.role || "zone"}</small>
                      </div>
                      <button
                        className="button small"
                        disabled={busy || !hasEvidence}
                        onClick={() =>
                          void act(
                            "accept_suggested_zone",
                            [camera.id, z.id, 5],
                            "Suggested zone accepted",
                          )
                        }
                      >
                        Accept
                        <Check size={14} />
                      </button>
                      <button
                        className="button small"
                        onClick={() => {
                          if (draft)
                            void act(
                              "update_scene_context",
                              [
                                camera.id,
                                {
                                  ...draft,
                                  zones: draft.zones?.filter(
                                    (v) => v.id !== z.id,
                                  ),
                                },
                              ],
                              "Suggestion ignored",
                            );
                        }}
                      >
                        Ignore
                      </button>
                    </div>
                  ))}
                </div>
                <div className="divider" />
                <p className="muted">
                  Reviewing a scene does not start monitoring. Use the workspace
                  monitoring control when ready.
                </p>
              </>
            )}
            {tab === "detectors" && (
              <>
                <div className="section-heading">
                  <h3>What should this camera watch for?</h3>
                  <p>Detector changes apply on the next monitoring start.</p>
                </div>
                {["Security", "Safety / HSE"].map((group) => (
                  <section key={group}>
                    <h4>{group}</h4>
                    {DETECTORS.filter((d) => d.group === group).map((d) => (
                      <div className="setting-row" key={d.key}>
                        <div>
                          <strong>{d.name}</strong>
                          <small>{d.detail}</small>
                        </div>
                        <Toggle
                          label={d.name}
                          checked={Boolean(camera[d.key])}
                          disabled={busy}
                          onChange={() =>
                            void act(
                              "set_camera_rules",
                              [camera.id, { [d.key]: !camera[d.key] }],
                              `${d.name} updated; restart monitoring to apply`,
                            )
                          }
                        />
                      </div>
                    ))}
                  </section>
                ))}
                {Object.keys(presets).length > 0 && (
                  <label>
                    Rule preset
                    <select
                      defaultValue=""
                      onChange={(e) => {
                        const value = presets[e.target.value];
                        void act(
                          "set_camera_rules",
                          [
                            camera.id,
                            {
                              config:
                                typeof value === "string"
                                  ? value
                                  : value.config || value.path,
                            },
                          ],
                          "Preset updated",
                        );
                      }}
                    >
                      <option value="" disabled>
                        Select a preset
                      </option>
                      {Object.keys(presets).map((k) => (
                        <option key={k}>{k}</option>
                      ))}
                    </select>
                  </label>
                )}
              </>
            )}
            {tab === "rules" && (
              <>
                <div className="section-heading">
                  <h3>Describe what matters.</h3>
                  <p>Define a visible condition to watch for in this camera.</p>
                </div>
                <div className="scanner-status">
                  <span className={`status-dot ${scanner ? "" : "sample"}`} />
                  {scanner
                    ? scanner.last_outcome || "Scanner active"
                    : english.demo
                      ? "Demo rules only; no AI scanner connected"
                      : "No scanner heartbeat yet"}
                  {scanner && (
                    <small>
                      {scanner.scans} scans · {scanner.hits} matches ·{" "}
                      {scanner.errors} errors
                    </small>
                  )}
                </div>
                {custom.map((r, i) => (
                  <div className="rule-item" key={i}>
                    <div>
                      <p>{r.question}</p>
                      <small>Saved rule · dwell setting {r.dwell}s</small>
                    </div>
                    <button
                      className="icon-button"
                      title="Remove rule"
                      aria-label={`Remove rule ${i + 1}`}
                      disabled={busy}
                      onClick={() =>
                        void act(
                          "remove_custom_rule",
                          [camera.id, r.question],
                          "Rule removed",
                        )
                      }
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
                {camera.custom_threats?.map((r) => (
                  <div className="rule-item" key={r.name}>
                    <div>
                      <strong>{r.name}</strong>
                      <p>{r.description}</p>
                      <small>Config-authored rule</small>
                    </div>
                  </div>
                ))}
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    void act(
                      "add_custom_rule",
                      [camera.id, question.trim(), dwell],
                      "English rule saved",
                    );
                  }}
                >
                  <label>
                    Watch condition
                    <textarea
                      required
                      minLength={5}
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="Alert when someone enters the loading bay carrying a ladder."
                      rows={3}
                    />
                  </label>
                  <label>
                    Dwell setting (seconds)
                    <input
                      type="number"
                      min="1"
                      max="86400"
                      value={dwell}
                      onChange={(e) => setDwell(Number(e.target.value))}
                    />
                  </label>
                  <p className="field-note">
                    The current English scanner checks individual images; this
                    setting is not proof of continuous duration. Use a zone rule
                    for measured loitering.
                  </p>
                  <button
                    className="button primary"
                    disabled={busy || question.trim().length < 5}
                  >
                    <Plus size={16} />
                    Add watch condition
                  </button>
                </form>
              </>
            )}
            {tab === "zones" && (
              <ZoneEditor
                camera={camera}
                api={api}
                onSaved={() => {
                  void onChange();
                  notify("Zone configuration saved");
                }}
              />
            )}
          </>
        )}
      </fieldset>
    </>
  );
}
