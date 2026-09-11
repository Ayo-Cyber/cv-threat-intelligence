import { useEffect, useMemo, useState } from "react";
import { MapPin, Plus, Save, Trash2 } from "lucide-react";
import type { Auth, Camera, Hierarchy, Mode, Transport } from "../lib/types";
import { decodeLocationId, locationIdToken } from "../lib/hierarchy";
import { Notice, Spinner } from "./common";

export default function LocationManager({
  api,
  mode,
  auth,
  hierarchy,
  cameras,
  onChange,
  notify,
}: {
  api: Transport;
  mode: Mode;
  auth: Auth;
  hierarchy: Hierarchy;
  cameras: Camera[];
  onChange: () => Promise<void>;
  notify: (message: string) => void;
}) {
  const canConfigureSite =
    mode === "engine" && auth.permissions.includes("configure_site");
  const canConfigureCameras =
    mode === "engine" && auth.permissions.includes("configure_cameras");
  const [organizationName, setOrganizationName] = useState(
    hierarchy.organization.name,
  );
  const [branchNames, setBranchNames] = useState<Record<string, string>>({});
  const [branchName, setBranchName] = useState("");
  const [areaName, setAreaName] = useState("");
  const [areaBranch, setAreaBranch] = useState(
    hierarchy.branches[0] ? locationIdToken(hierarchy.branches[0].id) : "",
  );
  const [cameraId, setCameraId] = useState(cameras[0]?.id || "");
  const [cameraBranch, setCameraBranch] = useState(
    hierarchy.branches[0] ? locationIdToken(hierarchy.branches[0].id) : "",
  );
  const [cameraArea, setCameraArea] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const hierarchyNameSignature = `${hierarchy.organization.name}|${hierarchy.branches
    .map((branch) => `${branch.id}:${branch.name}`)
    .join("|")}`;

  useEffect(() => {
    setOrganizationName(hierarchy.organization.name);
    setBranchNames(
      Object.fromEntries(
        hierarchy.branches.map((branch) => [branch.id, branch.name]),
      ),
    );
  }, [hierarchyNameSignature]);

  useEffect(() => {
    if (
      !hierarchy.branches.some(
        (branch) => branch.id === decodeLocationId(areaBranch),
      )
    )
      setAreaBranch(
        hierarchy.branches[0] ? locationIdToken(hierarchy.branches[0].id) : "",
      );
    if (
      !hierarchy.branches.some(
        (branch) => branch.id === decodeLocationId(cameraBranch),
      )
    ) {
      setCameraBranch(
        hierarchy.branches[0] ? locationIdToken(hierarchy.branches[0].id) : "",
      );
      setCameraArea("");
    }
    if (!cameras.some((camera) => camera.id === cameraId))
      setCameraId(cameras[0]?.id || "");
  }, [areaBranch, cameraBranch, cameraId, cameras, hierarchy.branches]);

  const placementAreas = useMemo(
    () =>
      hierarchy.branches.find(
        (branch) => branch.id === decodeLocationId(cameraBranch),
      )?.areas || [],
    [cameraBranch, hierarchy.branches],
  );

  async function run(method: string, args: unknown[], successMessage: string) {
    setBusy(true);
    setError("");
    try {
      await api.invoke(method, args);
      await onChange();
      notify(successMessage);
      return true;
    } catch (caught) {
      setError((caught as Error).message);
      return false;
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="settings-section location-section">
      <div>
        <h2>Locations</h2>
        <p>Organization, branches, areas, and camera placement.</p>
      </div>
      <div className="location-manager">
        {mode === "demo" && (
          <Notice>
            Sample location hierarchy. Demo changes never write to the local
            engine.
          </Notice>
        )}
        {error && <Notice error>{error}</Notice>}

        <div className="location-organization">
          <span className="eyebrow">ORGANIZATION</span>
          {canConfigureSite ? (
            <form
              className="compact-form"
              onSubmit={(event) => {
                event.preventDefault();
                void run(
                  "update_organization",
                  [
                    {
                      id: hierarchy.organization.id,
                      name: organizationName.trim(),
                    },
                  ],
                  "Organization updated",
                );
              }}
            >
              <label>
                Organization name
                <input
                  required
                  value={organizationName}
                  onChange={(event) => setOrganizationName(event.target.value)}
                />
              </label>
              <button
                className="button"
                disabled={busy || !organizationName.trim()}
              >
                {busy ? <Spinner /> : <Save size={16} />}
                Save organization
              </button>
            </form>
          ) : (
            <strong>{hierarchy.organization.name}</strong>
          )}
        </div>

        <div className="location-branches">
          {hierarchy.branches.map((branch) => {
            const cameraCount = branch.areas.reduce(
              (count, area) => count + area.cameras.length,
              0,
            );
            return (
              <div className="location-branch" key={branch.id}>
                <div className="location-branch-head">
                  {canConfigureCameras ? (
                    <form
                      className="compact-form"
                      onSubmit={(event) => {
                        event.preventDefault();
                        void run(
                          "update_branch",
                          [branch.id, { name: branchNames[branch.id]?.trim() }],
                          "Branch updated",
                        );
                      }}
                    >
                      <label>
                        Branch name
                        <input
                          aria-label={`Branch name: ${branch.name}`}
                          required
                          value={branchNames[branch.id] ?? branch.name}
                          onChange={(event) =>
                            setBranchNames((current) => ({
                              ...current,
                              [branch.id]: event.target.value,
                            }))
                          }
                        />
                      </label>
                      <button
                        className="icon-button"
                        title={`Save ${branch.name}`}
                        aria-label={`Save ${branch.name}`}
                        disabled={
                          busy || !(branchNames[branch.id] ?? "").trim()
                        }
                      >
                        <Save size={16} />
                      </button>
                    </form>
                  ) : (
                    <strong>{branch.name}</strong>
                  )}
                  <small>
                    {branch.areas.length} areas · {cameraCount} cameras
                  </small>
                  {canConfigureCameras && (
                    <button
                      className="icon-button"
                      title={`Delete ${branch.name}`}
                      disabled={busy || branch.areas.length > 0}
                      aria-label={`Delete ${branch.name}`}
                      onClick={() => {
                        if (confirm(`Delete empty branch ${branch.name}?`))
                          void run(
                            "remove_branch",
                            [branch.id],
                            "Branch deleted",
                          );
                      }}
                    >
                      <Trash2 size={16} />
                    </button>
                  )}
                </div>
                <div className="location-area-list">
                  {branch.areas.map((area) => (
                    <div className="setting-row" key={area.id}>
                      <div>
                        <strong>{area.name}</strong>
                        <small>{area.cameras.length} cameras</small>
                      </div>
                      <MapPin size={16} />
                    </div>
                  ))}
                  {!branch.areas.length && (
                    <p className="field-note">No areas in this branch.</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {!!hierarchy.unassigned_cameras.length && (
          <div className="location-unassigned">
            <strong>Unassigned</strong>
            <small>{hierarchy.unassigned_cameras.length} cameras</small>
            <p className="field-note">
              {hierarchy.unassigned_cameras
                .map((camera) => camera.id)
                .join(", ")}
            </p>
          </div>
        )}

        {canConfigureCameras && (
          <>
            <div className="divider" />
            <form
              className="location-create"
              onSubmit={(event) => {
                event.preventDefault();
                void run(
                  "create_branch",
                  [
                    {
                      id: branchName
                        .trim()
                        .toLowerCase()
                        .replace(/[^a-z0-9]+/g, "-")
                        .replace(/^-|-$/g, ""),
                      name: branchName.trim(),
                    },
                  ],
                  "Branch created",
                ).then((ok) => ok && setBranchName(""));
              }}
            >
              <label>
                New branch
                <input
                  required
                  value={branchName}
                  onChange={(event) => setBranchName(event.target.value)}
                  placeholder="Ikeja Outlet"
                />
              </label>
              <button className="button" disabled={busy || !branchName.trim()}>
                <Plus size={16} />
                Add branch
              </button>
            </form>

            <form
              className="location-create"
              onSubmit={(event) => {
                event.preventDefault();
                void run(
                  "create_area",
                  [
                    {
                      id: areaName
                        .trim()
                        .toLowerCase()
                        .replace(/[^a-z0-9]+/g, "-")
                        .replace(/^-|-$/g, ""),
                      name: areaName.trim(),
                      branch_id: decodeLocationId(areaBranch),
                    },
                  ],
                  "Area created",
                ).then((ok) => ok && setAreaName(""));
              }}
            >
              <label>
                New area
                <input
                  required
                  value={areaName}
                  onChange={(event) => setAreaName(event.target.value)}
                  placeholder="Paint floor"
                />
              </label>
              <label>
                Branch
                <select
                  required
                  value={areaBranch}
                  onChange={(event) => setAreaBranch(event.target.value)}
                >
                  {hierarchy.branches.map((branch) => (
                    <option value={locationIdToken(branch.id)} key={branch.id}>
                      {branch.name}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="button"
                disabled={busy || !areaName.trim() || !areaBranch}
              >
                <Plus size={16} />
                Add area
              </button>
            </form>

            <form
              className="location-create"
              onSubmit={(event) => {
                event.preventDefault();
                void run(
                  "assign_camera_area",
                  [cameraId, decodeLocationId(cameraArea)],
                  "Camera location updated",
                );
              }}
            >
              <label>
                Camera
                <select
                  required
                  value={cameraId}
                  onChange={(event) => setCameraId(event.target.value)}
                >
                  {cameras.map((camera) => (
                    <option value={camera.id} key={camera.id}>
                      {camera.id}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Branch
                <select
                  required
                  value={cameraBranch}
                  onChange={(event) => {
                    setCameraBranch(event.target.value);
                    setCameraArea("");
                  }}
                >
                  {hierarchy.branches.map((branch) => (
                    <option value={locationIdToken(branch.id)} key={branch.id}>
                      {branch.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Area
                <select
                  required
                  value={cameraArea}
                  onChange={(event) => setCameraArea(event.target.value)}
                >
                  <option value="">Select area</option>
                  {placementAreas.map((area) => (
                    <option value={locationIdToken(area.id)} key={area.id}>
                      {area.name}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="button primary"
                disabled={busy || !cameraId || !cameraArea}
              >
                <MapPin size={16} />
                Assign camera
              </button>
            </form>
          </>
        )}
      </div>
    </section>
  );
}
