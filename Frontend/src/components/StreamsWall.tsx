import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Maximize2,
  Minimize2,
  Search,
  X,
} from "lucide-react";
import type { Camera, Hierarchy, Mode, Transport } from "../lib/types";
import {
  ALL_LOCATIONS,
  areaOptions,
  branchOptions,
  decodeLocationSelection,
  encodeLocationSelection,
  filterCameras,
  reconcileAreaSelection,
  type LocationSelection,
} from "../lib/hierarchy";
import {
  reconcileFocusedStream,
  subscribedStreamIds,
  useVisibleStreams,
} from "../hooks/useVisibleStreams";
import CameraStream from "./CameraStream";
import { CameraMedia, Empty } from "./common";

const DENSITIES = [4, 9, 16] as const;

function healthKind(camera: Camera): "healthy" | "degraded" | "offline" {
  const state = String(camera.state || "").toLowerCase();
  if (["offline", "failed", "error"].includes(state)) return "offline";
  if (["degraded", "stalled", "reconnecting"].includes(state))
    return "degraded";
  return "healthy";
}

export default function StreamsWall({
  hierarchy,
  cameras,
  api,
  mode,
  running,
  onExit,
}: {
  hierarchy: Hierarchy;
  cameras: Camera[];
  api: Transport;
  mode: Mode;
  running: boolean;
  onExit: () => void;
}) {
  const [branch, setBranch] = useState<LocationSelection>(ALL_LOCATIONS);
  const [area, setArea] = useState<LocationSelection>(ALL_LOCATIONS);
  const [query, setQuery] = useState("");
  const [density, setDensity] = useState<(typeof DENSITIES)[number]>(4);
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const [toolbarHidden, setToolbarHidden] = useState(false);
  const [toolbarFocused, setToolbarFocused] = useState(false);
  const [fullscreen, setFullscreen] = useState(
    Boolean(document.fullscreenElement),
  );
  const wall = useRef<HTMLElement>(null);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const filtered = useMemo(
    () => filterCameras(cameras, hierarchy, branch, area, query),
    [area, branch, cameras, hierarchy, query],
  );
  const cameraById = useMemo(
    () => new Map(cameras.map((camera) => [camera.id, camera])),
    [cameras],
  );
  const areasById = useMemo(
    () =>
      new Map(
        hierarchy.branches.flatMap((item) =>
          item.areas.map((areaItem) => [areaItem.id, areaItem.name] as const),
        ),
      ),
    [hierarchy],
  );
  const branchChoices = branchOptions(cameras, hierarchy);
  const areaChoices = areaOptions(cameras, hierarchy, branch);
  const filteredIds = filtered.map((camera) => camera.id);
  const {
    activeIds,
    visibleIds,
    prefetchIds,
    page,
    pageCount,
    setPage,
    observeWall,
  } = useVisibleStreams(filteredIds, density);
  const effectiveFocusedId = reconcileFocusedStream(focusedId, filteredIds);
  const displayedIds = effectiveFocusedId ? [effectiveFocusedId] : visibleIds;
  const subscribedIds = new Set(
    subscribedStreamIds(effectiveFocusedId, activeIds),
  );

  useEffect(() => {
    if (focusedId && !effectiveFocusedId) setFocusedId(null);
  }, [effectiveFocusedId, focusedId]);

  const showToolbar = useCallback(() => {
    setToolbarHidden(false);
    if (hideTimer.current) clearTimeout(hideTimer.current);
    if (!toolbarFocused) {
      hideTimer.current = setTimeout(() => setToolbarHidden(true), 3000);
    }
  }, [toolbarFocused]);

  useEffect(() => {
    showToolbar();
    return () => {
      if (hideTimer.current) clearTimeout(hideTimer.current);
    };
  }, [showToolbar]);

  useEffect(() => {
    const onFullscreenChange = () =>
      setFullscreen(document.fullscreenElement === wall.current);
    document.addEventListener("fullscreenchange", onFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", onFullscreenChange);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      showToolbar();
      if (event.key !== "Escape") return;
      if (effectiveFocusedId) {
        event.preventDefault();
        event.stopPropagation();
        setFocusedId(null);
      } else if (document.fullscreenElement) {
        event.preventDefault();
        event.stopPropagation();
        void document.exitFullscreen();
      } else {
        onExit();
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [effectiveFocusedId, onExit, showToolbar]);

  const toggleFullscreen = async () => {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await wall.current?.requestFullscreen();
  };
  const health = filtered.reduce(
    (counts, camera) => {
      counts[healthKind(camera)] += 1;
      return counts;
    },
    { healthy: 0, degraded: 0, offline: 0 },
  );

  return (
    <section
      className="streams-wall"
      ref={wall}
      role="region"
      aria-label="Streams-only camera wall"
      data-fullscreen={fullscreen}
      onPointerMove={showToolbar}
      onPointerDown={showToolbar}
    >
      <div
        className="streams-toolbar"
        role="toolbar"
        aria-label="Camera wall controls"
        data-hidden={toolbarHidden}
        onFocusCapture={() => {
          setToolbarFocused(true);
          setToolbarHidden(false);
          if (hideTimer.current) clearTimeout(hideTimer.current);
        }}
        onBlurCapture={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node)) {
            setToolbarFocused(false);
          }
        }}
      >
        <strong className="streams-identity">
          <span className="status-dot" />
          {hierarchy.organization.name || "ARGUS"}
        </strong>
        <div className="streams-filters">
          <select
            aria-label="Wall branch"
            value={encodeLocationSelection(branch)}
            onChange={(event) => {
              const next = decodeLocationSelection(event.target.value);
              if (!next) return;
              setBranch(next);
              setArea((current) =>
                reconcileAreaSelection(hierarchy, next, current),
              );
            }}
          >
            {branchChoices.map((option) => (
              <option
                key={encodeLocationSelection(option.selection)}
                value={encodeLocationSelection(option.selection)}
              >
                {option.name} ({option.count})
              </option>
            ))}
          </select>
          <select
            aria-label="Wall area"
            value={encodeLocationSelection(area)}
            onChange={(event) => {
              const next = decodeLocationSelection(event.target.value);
              if (next) setArea(next);
            }}
          >
            {areaChoices.map((option) => (
              <option
                key={encodeLocationSelection(option.selection)}
                value={encodeLocationSelection(option.selection)}
              >
                {option.name} ({option.count})
              </option>
            ))}
          </select>
          <label className="streams-search">
            <Search size={14} />
            <input
              aria-label="Search streams"
              placeholder="Search cameras"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
        </div>
        <div className="streams-density" role="group" aria-label="Wall density">
          {DENSITIES.map((size) => (
            <button
              key={size}
              aria-label={`Show ${size} cameras`}
              title={`Show ${size} cameras`}
              aria-pressed={density === size}
              onClick={() => {
                setDensity(size);
                setFocusedId(null);
              }}
            >
              {size}
            </button>
          ))}
        </div>
        <div className="streams-pager">
          <button
            className="icon-button"
            aria-label="Previous camera page"
            title="Previous page"
            disabled={page <= 1}
            onClick={() => setPage(page - 1)}
          >
            <ChevronLeft size={17} />
          </button>
          <span>
            {page} / {pageCount}
          </span>
          <button
            className="icon-button"
            aria-label="Next camera page"
            title="Next page"
            disabled={page >= pageCount}
            onClick={() => setPage(page + 1)}
          >
            <ChevronRight size={17} />
          </button>
        </div>
        <div className="streams-health" aria-label="Camera health counts">
          <span title="Healthy cameras">
            <i className="healthy" />
            {health.healthy}
          </span>
          <span title="Degraded cameras">
            <i className="degraded" />
            {health.degraded}
          </span>
          <span title="Offline cameras">
            <i className="offline" />
            {health.offline}
          </span>
        </div>
        <div className="streams-actions">
          <button
            className="icon-button"
            aria-label={fullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            title={fullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            onClick={() => void toggleFullscreen()}
          >
            {fullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          </button>
          <button
            className="icon-button"
            aria-label="Exit streams wall"
            title="Exit streams wall"
            onClick={onExit}
          >
            <X size={19} />
          </button>
        </div>
      </div>

      <div
        className={`streams-stage density-${density} ${effectiveFocusedId ? "is-focused" : ""}`}
        ref={observeWall}
      >
        {displayedIds.map((id) => {
          const camera = cameraById.get(id);
          if (!camera) return null;
          return (
            <article className="streams-tile" key={camera.id}>
              {mode === "demo" ? (
                <CameraMedia camera={camera} paused={!running} />
              ) : (
                <CameraStream
                  camera={camera}
                  api={api}
                  active={subscribedIds.has(camera.id)}
                />
              )}
              <div className="streams-caption">
                <span>
                  <strong>{camera.id}</strong>
                  <small>
                    {areasById.get(camera.area_id || "") || "Unassigned"}
                  </small>
                </span>
                <button
                  className="icon-button"
                  aria-label={`Focus ${camera.id}`}
                  title={`Focus ${camera.id}`}
                  onClick={() => setFocusedId(camera.id)}
                >
                  <Maximize2 size={16} />
                </button>
              </div>
            </article>
          );
        })}
        {!displayedIds.length && (
          <Empty title="No matching cameras">
            Change the branch, area, or search filter.
          </Empty>
        )}
      </div>

      {mode === "engine" && !effectiveFocusedId && (
        <div className="streams-prefetch" aria-hidden="true">
          {prefetchIds.map((id) => {
            const camera = cameraById.get(id);
            return camera ? (
              <CameraStream
                key={id}
                camera={camera}
                api={api}
                active={subscribedIds.has(id)}
              />
            ) : null;
          })}
        </div>
      )}
    </section>
  );
}
