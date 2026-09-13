# Chi Pilot Motion Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reliable normal movement telemetry, simultaneous-person movement candidates, and optional person overlays for Chi scenarios 4 and 5.

**Architecture:** A new detector module turns Supervision ByteTrack boxes into smoothed per-track motion snapshots, then a separate aggregate detector recognizes simultaneous movement. The engine always tracks internally; authenticated raw and annotated stream variants let each operator control visual noise independently from detection.

**Tech Stack:** Python 3.9, Ultralytics YOLO, Supervision ByteTrack, OpenCV, FastAPI, Electron, React, TypeScript, Vitest, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-13-chi-pilot-motion-scenarios-design.md`

## Global Constraints

- Scenario 4 is telemetry and must not generate an alert by itself.
- Scenario 5 is distinct from crowd formation and does not require proximity.
- Detection continues when overlays are hidden.
- Global overlay preference is persisted per operator; camera overrides are session-only.
- Tracking overlays default to hidden.
- Raw and annotated streams use the same authentication boundary.
- No new model or training dependency.

---

### Task 1: Build Reusable Person Motion State

**Files:**
- Create: `cvti/detector/person_motion.py`
- Create: `tests/test_person_motion.py`

**Interfaces:**
- Produces: `PersonMotion(track_id, bbox, speed_ratio, moving, moving_seconds, zone_names)`
- Produces: `PersonMotionTracker.update(people: Sequence[tuple[int, float, float, float, float]], timestamp, frame_shape, zones_by_track=None) -> list[PersonMotion]`
- Produces: `SimultaneousMovementDetector.update(motions, timestamp) -> dict | None`

- [ ] **Step 1: Write failing speed and jitter tests**

```python
def test_sustained_displacement_becomes_moving():
    tracker = PersonMotionTracker(enter_speed_ratio=0.05, exit_speed_ratio=0.02,
                                  min_track_seconds=0.4)
    assert not tracker.update([person(1, 10)], 0.0, FRAME)[0].moving
    result = tracker.update([person(1, 70)], 0.5, FRAME)[0]
    assert result.moving

def test_bbox_jitter_stays_stationary():
    tracker = PersonMotionTracker(enter_speed_ratio=0.05, exit_speed_ratio=0.02,
                                  min_track_seconds=0.4)
    for index, x in enumerate((10, 11, 9, 12, 10)):
        motion = tracker.update([person(1, x)], index * 0.2, FRAME)[0]
    assert not motion.moving
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_person_motion.py -q`

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement smoothing, hysteresis, and expiry**

Normalize center velocity by frame diagonal, apply an EMA, require minimum track age before entering moving state, use the lower exit threshold to reset, and expire tracks after a short absence.

```python
@dataclass(frozen=True)
class PersonMotion:
    track_id: int
    bbox: tuple[float, float, float, float]
    speed_ratio: float
    moving: bool
    moving_seconds: float
    zone_names: tuple[str, ...]

def update(
    self,
    people: Sequence[tuple[int, float, float, float, float]],
    timestamp: float,
    frame_shape: tuple[int, int],
    zones_by_track: Mapping[int, Sequence[str]] | None = None,
) -> list[PersonMotion]:
    """Return current smoothed motion state and expire stale tracks."""
```

- [ ] **Step 4: Write failing simultaneous-movement tests**

Assert that two sustained moving tracks fire one candidate, one moving track does not, a stationary cluster does not, spread-out moving people do fire, and the latch resets only after count drops.

- [ ] **Step 5: Implement the aggregate detector and commit**

```python
def update(self, motions: Sequence[PersonMotion], timestamp: float) -> dict | None:
    qualifying = [motion for motion in motions if motion.moving]
    if len(qualifying) < self.min_people:
        self._active_since = None
        self._latched = False
        return None
    if self._active_since is None:
        self._active_since = timestamp
    if self._latched or timestamp - self._active_since < self.persistence_seconds:
        return None
    self._latched = True
    return movement_event_metadata(qualifying)
```

Run: `./.venv/bin/python -m pytest tests/test_person_motion.py -q`

Commit: `feat(motion): classify tracked movement and simultaneous activity`

---

### Task 2: Integrate Scenarios 4 And 5 Into The Camera Pipeline

**Files:**
- Modify: `cvti/serving/camera.py:130-220,337-460,581-655`
- Modify: `cvti/event_adapters.py`
- Modify: `cvti/verification/frame_select.py:25-42`
- Modify: `cvti/verification/gate.py:135-160`
- Modify: `cvti/app/console_backend.py` detector toggle definitions
- Modify: `Frontend/src/lib/types.ts` detector definitions
- Create: `configs/chi_pilot_v1.json`
- Modify: `tests/test_detector_toggles.py`
- Modify: `tests/test_serving.py`
- Modify: `tests/test_gate_evidence_quality.py`

**Interfaces:**
- Consumes: `PersonMotionTracker` and `SimultaneousMovementDetector` from Task 1.
- Produces: `PerCameraState._motion_overlays` and `RawEvent(detector="multiple_people_moving")`.

- [ ] **Step 1: Write failing configuration and zone-filtering tests**

Build a camera with `normal_movement`, `multiple_people_moving`, speed thresholds, minimum people, persistence, and permitted zones. Assert every value reaches `PerCameraState` and invalid thresholds are rejected with the camera ID. Feed one moving track inside an allowed zone and one outside; assert only the allowed track is overlaid and counted. Repeat without `permitted_movement_zones` and assert both tracks count.

- [ ] **Step 2: Run tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_detector_toggles.py tests/test_serving.py -q`

- [ ] **Step 3: Add state fields and zone-aware motion processing**

Update movement after ByteTrack and zone membership are known. Store overlays only for moving tracks inside permitted zones. Do not emit a `RawEvent` for `normal_movement`.

```python
motions = self._motion_tracker.update(
    tracked_people,
    timestamp,
    frame.shape[:2],
    zones_by_track=zones_by_track,
)
self._motion_overlays = [
    motion_overlay(motion)
    for motion in motions
    if motion.moving and self._movement_zone_allows(motion.zone_names)
]
```

- [ ] **Step 4: Write failing scenario-5 pipeline tests**

Assert two qualifying tracks create one `multiple_people_moving` event, stationary people create none, the event contains all track IDs and a group box, and latching prevents per-frame duplicates.

- [ ] **Step 5: Route scenario 5 through rules and verification**

Add the Chi rule with a configurable priority, three-frame evidence selection, and a gate question that asks about simultaneous visible movement rather than crowd density or panic.

```python
RawEvent(
    detector="multiple_people_moving",
    confidence=event["confidence"],
    bbox=event["group_bbox"],
    track_id=None,
    extra={"track_ids": event["track_ids"], "motions": event["motions"]},
)
```

- [ ] **Step 6: Expose detector controls and commit**

Add scenario toggles to the existing backend/frontend detector catalogs. Run:

`./.venv/bin/python -m pytest tests/test_person_motion.py tests/test_detector_toggles.py tests/test_serving.py tests/test_gate_evidence_quality.py -q`

Commit: `feat(pilot): wire normal and simultaneous movement scenarios`

---

### Task 3: Serve Authenticated Raw And Tracking Stream Variants

**Files:**
- Modify: `cvti/serving/frame_publisher.py:41-148,190-250`
- Modify: `cvti/serving/pipeline.py:395-485,880-900`
- Modify: `cvti/api/app.py:276-318`
- Modify: `Frontend/electron/api-client.ts:155-170`
- Modify: `Frontend/electron/bridge-transport.ts:60-90`
- Modify: `Frontend/src/lib/whep.ts:130-165`
- Modify: `Frontend/src/lib/types.ts:88-105`
- Modify: `tests/test_frame_publisher.py`
- Modify: `tests/test_serving.py`
- Modify: `tests/test_engine_api.py`
- Modify: `Frontend/tests/api-client.test.ts`
- Modify: `Frontend/tests/bridge-transport.test.ts`
- Modify: `Frontend/tests/whep.test.ts`

**Interfaces:**
- Produces: `camera_stream(camera_id, tracking=False)`.
- Produces: authenticated MJPEG `/stream/{camera_id}?tracking=1&token=...`.
- Consumes: `PerCameraState._motion_overlays` from Task 2.

- [ ] **Step 1: Write failing publisher-variant tests**

Connect one raw and one tracking viewer. Publish a frame with a moving-track overlay. Assert raw JPEG pixels remain unchanged, annotated JPEG contains the box, viewer counts are released on disconnect, and a bad token receives 401 for both variants.

- [ ] **Step 2: Run tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_frame_publisher.py -q`

- [ ] **Step 3: Implement on-demand annotated publishing**

Track overlay viewers separately. Keep raw JPEG as the canonical frame and generate/cache an annotated JPEG only while an overlay viewer exists. Extend overlay records with track ID, box, label, and semantic colour.

```python
@dataclass(frozen=True)
class FrameOverlay:
    track_id: int
    bbox: tuple[int, int, int, int]
    label: str
    colour: tuple[int, int, int]

def publish(self, camera_id: str, frame, overlays: Sequence[FrameOverlay] = ()) -> None:
    self._raw_frames[camera_id] = encode_jpeg(frame)
    if self._tracking_viewers.get(camera_id, 0):
        self._tracking_frames[camera_id] = encode_jpeg(draw_overlays(frame, overlays))
```

- [ ] **Step 4: Feed cached overlays into smooth publishing**

Replace the current raw-only smooth publish call with the latest motion overlays. Preserve view-only cameras as raw glass and preserve detection cadence independence.

- [ ] **Step 5: Write failing API and Electron transport tests**

Assert `tracking=true` returns the annotated MJPEG descriptor even when WebRTC is available, `tracking=false` still prefers WebRTC, and both native API and legacy bridge sanitize and forward the boolean without exposing tokens.

- [ ] **Step 6: Implement API and transport contracts**

Pass the second `camera_stream` argument through the API client and legacy bridge. When tracking is requested, select the local MJPEG publisher; otherwise retain the existing WebRTC-first response.

```typescript
camera_stream(cameraId: string, tracking = false): Promise<CameraStreamDescriptor>
```

```python
@app.get("/api/v1/cameras/{camera_id}/stream")
def camera_stream(camera_id: str, tracking: bool = False):
    return stream_descriptor(camera_id, prefer_webrtc=not tracking, tracking=tracking)
```

- [ ] **Step 7: Run transport tests and commit**

Run: `./.venv/bin/python -m pytest tests/test_frame_publisher.py tests/test_engine_api.py tests/test_serving.py -q`

Run: `cd Frontend && npm test -- api-client.test.ts bridge-transport.test.ts whep.test.ts`

Commit: `feat(streams): add optional authenticated tracking overlays`

---

### Task 4: Add Global And Per-Camera Overlay Controls

**Files:**
- Create: `Frontend/src/lib/tracking-overlay.ts`
- Modify: `Frontend/src/App.tsx`
- Modify: `Frontend/src/components/StreamsWall.tsx`
- Modify: `Frontend/src/components/CameraStream.tsx`
- Modify: `Frontend/src/styles.css`
- Create: `Frontend/tests/tracking-overlay.test.ts`
- Modify: `Frontend/tests/streams-wall.test.ts`
- Modify: `Frontend/tests/ui.spec.ts`

**Interfaces:**
- Produces: `TrackingPreference = "global" | "show" | "hide"`.
- Produces: `trackingVisible(globalVisible, cameraOverride) -> boolean`.
- Consumes: `camera_stream(camera_id, tracking)` from Task 3.

- [ ] **Step 1: Write failing preference tests**

Assert the global default is false, persisted global true survives remount, `show` overrides global false, `hide` overrides global true, and overrides disappear after a new session.

- [ ] **Step 2: Run tests and confirm RED**

Run: `cd Frontend && npm test -- tracking-overlay.test.ts streams-wall.test.ts`

- [ ] **Step 3: Implement the preference helper and controls**

Add a toolbar toggle with an eye/scan icon and accessible `Show tracking` label. Add a compact camera menu with `Use global`, `Show`, and `Hide`. Do not place text controls over the video content; keep them in the existing tile action area with tooltips.

```typescript
export type TrackingPreference = "global" | "show" | "hide";

export function trackingVisible(
  globalVisible: boolean,
  cameraOverride: TrackingPreference,
): boolean {
  if (cameraOverride === "show") return true;
  if (cameraOverride === "hide") return false;
  return globalVisible;
}
```

Persist the global value under `argus:tracking-overlay:${operatorId}`. Keep camera overrides in React state only so restarting the desktop session clears them.

- [ ] **Step 4: Remount only the affected stream**

Pass the resolved boolean into `CameraStream`, include it in the stream-resolution effect dependencies, and request `camera_stream(camera.id, tracking)`. Changing the global preference remounts visible streams; changing one override remounts only that camera.

```typescript
useEffect(() => {
  void resolveCameraStream(camera.id, tracking);
  return stopResolvedStream;
}, [camera.id, tracking]);
```

- [ ] **Step 5: Add UI workflow coverage**

Use the real component and transport fixture to assert the toolbar toggle, camera override precedence, default clean view, persistence, and that monitoring remains running while visibility changes.

- [ ] **Step 6: Run frontend checks and commit**

Run: `cd Frontend && npm test`

Run: `cd Frontend && npm run build`

Commit: `feat(frontend): add operator tracking-overlay controls`

---

### Task 5: Validate Scenarios 4 And 5 End To End

**Files:**
- Modify: `docs/PROJECT_CONTEXT.md`
- Modify: `docs/CHI_PILOT_TESTING.md`
- Add test fixtures only when redistribution rights allow them.

**Interfaces:**
- Produces: repeatable Chi pilot commands, labeled intervals, and measured results.

- [ ] **Step 1: Document positive and negative clips**

Include one-person permitted movement, stationary jitter, two simultaneous movers, stationary crowd, spread-out movers, temporary occlusion, and zone-boundary crossing.

- [ ] **Step 2: Run focused backend and frontend suites**

Run: `./.venv/bin/python -m pytest tests/test_person_motion.py tests/test_retail_zones.py tests/test_serving.py tests/test_frame_publisher.py tests/test_engine_api.py -q`

Run: `cd Frontend && npm test && npm run build`

- [ ] **Step 3: Run full regression**

Run: `./.venv/bin/python -m pytest`

- [ ] **Step 4: Run a local pilot replay**

Start the engine with a Chi site config, enable tracking overlays, and record detection recall, ID switches, scenario-5 precision/recall, duplicate alerts, detection delay, and FPS with overlays hidden and shown.

- [ ] **Step 5: Commit documentation**

Commit: `docs(pilot): add Chi motion acceptance workflow`
