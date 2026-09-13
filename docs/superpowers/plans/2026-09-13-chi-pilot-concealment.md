# Chi Pilot Pocketing And Bagging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the production concealment timeline and activate per-person personal-bag grounding for Chi scenario 10.

**Architecture:** Keep `ConcealmentDetector` as the temporal candidate generator, but separate sampled pose observations from ordinary skipped video frames. Reuse the shared COCO detections for bags, associate bags spatially to pose tracks, and preserve the existing rules, Agent Mapper, and TrueSight final-verdict flow.

**Tech Stack:** Python 3.9, Ultralytics YOLO, Supervision, NumPy, pytest/unittest, existing FastAPI/serving pipeline.

**Spec:** `docs/superpowers/specs/2026-09-13-chi-pilot-concealment-design.md`

## Global Constraints

- No new model or training dependency.
- Keep `heavy_stride` performance behaviour.
- Trolleys and shopping baskets remain safe destinations.
- A candidate is temporal evidence, not a theft verdict.
- Existing standalone `bag_bboxes` callers remain compatible.
- Threshold changes require measured pilot evidence.

---

### Task 1: Preserve Concealment History Across Pose Sampling Gaps

**Files:**
- Modify: `cvti/retail/concealment.py:209-268`
- Modify: `cvti/serving/camera.py:464-492`
- Test: `tests/test_concealment.py`
- Test: `tests/test_heavy_models_earn_their_frames.py`

**Interfaces:**
- Produces: `ConcealmentDetector.expire(timestamp: float) -> None`
- Preserves: `ConcealmentDetector.update(pose_frames, timestamp, bag_bboxes=None)`
- Consumes: `PerCameraState.run_heavy` sampling decision.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_unsampled_gap_does_not_erase_concealment_history():
    det = ConcealmentDetector(state_grace_seconds=1.5)
    det.update([frame(0.0, (200.0, 110.0))], 0.0)
    det.expire(0.2)
    assert 1 in det._buffers

def test_stale_track_is_expired_after_grace():
    det = ConcealmentDetector(state_grace_seconds=1.0)
    det.update([frame(0.0, (200.0, 110.0))], 0.0)
    det.expire(1.01)
    assert 1 not in det._buffers
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_concealment.py -q`

Expected: failure because `state_grace_seconds` and `expire()` do not exist.

- [ ] **Step 3: Implement sampled-observation lifecycle**

Add `_last_seen: dict[int, float]`, update it for observed pose tracks, remove immediate deletion of absent IDs from `update()`, and implement `expire()` using `state_grace_seconds`. Expiry removes `_buffers`, `_over_threshold`, and `_last_seen` together.

```python
def expire(self, timestamp: float) -> None:
    stale = [
        track_id
        for track_id, seen_at in self._last_seen.items()
        if timestamp - seen_at > self.state_grace_seconds
    ]
    for track_id in stale:
        self._buffers.pop(track_id, None)
        self._over_threshold.pop(track_id, None)
        self._last_seen.pop(track_id, None)
```

- [ ] **Step 4: Add a failing production-stride regression test**

Drive `PerCameraState.process()` across a reach-then-waist sequence with `heavy_stride=2` and a deterministic pose provider. Assert that the concealment assessment reaches candidate state; the test must fail while skipped frames still call `update([])`.

- [ ] **Step 5: Integrate lifecycle into `PerCameraState`**

Call `self._conceal.update(...)` only when pose inference ran. Call `self._conceal.expire(timestamp)` every frame. Do not turn a skipped pose pass into an empty observation.

```python
if pose_ran:
    concealment = self._conceal.update(
        pose_frames,
        timestamp,
        bag_bboxes=bag_boxes,
    )
self._conceal.expire(timestamp)
```

- [ ] **Step 6: Run focused tests and commit**

Run: `./.venv/bin/python -m pytest tests/test_concealment.py tests/test_heavy_models_earn_their_frames.py tests/test_serving.py -q`

Commit: `fix(concealment): preserve motion history across pose stride`

---

### Task 2: Ground Personal Bags Per Person

**Files:**
- Modify: `cvti/retail/concealment.py:150-206,232-268`
- Modify: `cvti/serving/camera.py:50-68,464-492`
- Test: `tests/test_concealment.py`
- Test: `tests/test_serving.py`

**Interfaces:**
- Produces: `personal_bag_boxes(detections: Any) -> list[tuple[float, float, float, float]]`
- Produces: `bags_for_pose(frame: PoseFrame, bag_bboxes: list[tuple]) -> list[tuple]`
- Extends: `ConcealmentDetector.update(pose_frames, timestamp, bag_bboxes=None, bag_bboxes_by_track=None)`; a track-specific value takes precedence over the legacy global list.
- Extends: `ConcealmentAssessment.associated_bag: tuple[float, float, float, float] | None`.
- Consumes: shared YOLO/Supervision detections with COCO class IDs 24, 26, and 28.

- [ ] **Step 1: Write failing extraction and association tests**

Use real `sv.Detections` containing one person, one handbag, and one unrelated object. Assert extraction returns only the handbag. Create two `PoseFrame` boxes with one nearby bag and assert only the nearby person's score receives `f_bag > 0`; also assert a trolley-shaped ordinary object is never accepted as a personal bag.

- [ ] **Step 2: Run the tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_concealment.py -q`

Expected: failure because extraction and per-person association do not exist.

- [ ] **Step 3: Implement bag extraction and spatial association**

Extract boxes whose `class_id` belongs to `COCO_BAG_IDS`. For each pose frame, accept bags intersecting an expanded person box or within one torso/body scale. Pass only that subset into `_frame_features()`.

```python
COCO_BAG_IDS = frozenset({24, 26, 28})

def personal_bag_boxes(detections) -> list[tuple[float, float, float, float]]:
    return [
        tuple(map(float, box))
        for box, class_id in zip(detections.xyxy, detections.class_id)
        if int(class_id) in COCO_BAG_IDS
    ]

def bags_for_pose(frame: PoseFrame, bag_bboxes: list[tuple]) -> list[tuple]:
    return [box for box in bag_bboxes if bag_belongs_to_person(box, frame.bbox)]
```

Add `associated_bag` to `ConcealmentAssessment`, add `bag_bboxes_by_track` to `update()`, and select the bags for each observed pose with this compatibility helper:

```python
def _bags_for_track(
    track_id: int,
    bag_bboxes: list[tuple[float, float, float, float]] | None = None,
    bag_bboxes_by_track: dict[int, list[tuple[float, float, float, float]]] | None = None,
) -> list[tuple[float, float, float, float]]:
    if bag_bboxes_by_track is not None:
        return bag_bboxes_by_track.get(track_id, [])
    return bag_bboxes or []

associated_bag: tuple[float, float, float, float] | None = None
```
- [ ] **Step 4: Write a failing production-wiring test**

Process detections containing a COCO handbag and patch only the expensive pose forward pass. Assert the real concealment detector receives the associated bag for the matching pose track and not for a second shopper.

- [ ] **Step 5: Wire shared detections into concealment**

Extract bags before person-only use, pass them to `ConcealmentDetector.update()`, and avoid an additional YOLO invocation.

```python
bag_boxes = personal_bag_boxes(detections)
assessments = self._conceal.update(
    pose_frames,
    timestamp,
    bag_bboxes_by_track={
        pose.track_id: bags_for_pose(pose, bag_boxes) for pose in pose_frames
    },
)
```

- [ ] **Step 6: Run focused tests and commit**

Run: `./.venv/bin/python -m pytest tests/test_concealment.py tests/test_serving.py tests/test_heavy_models_earn_their_frames.py -q`

Commit: `feat(concealment): ground personal bags per tracked shopper`

---

### Task 3: Preserve Detector Cues Through Rules And TrueSight

**Files:**
- Modify: `cvti/event_adapters.py:93-116`
- Modify: `cvti/verification/gate.py:120-160`
- Test: `tests/test_concealment.py`
- Test: `tests/test_prompt_regression.py`
- Test: `tests/test_gate_evidence_quality.py`

**Interfaces:**
- Produces: concealment `RawEvent.extra` containing destination, score, components, reasons, limited, and associated bag.
- Preserves: three chronological concealment evidence frames plus subject crop.

- [ ] **Step 1: Write failing metadata, normal-action, and prompt tests**

Assert event metadata retains the component scores, reasons, limited-evidence flag, and associated bag. Add synthetic normal browsing, phone-to-pocket, clothing-adjustment, open-carry, and missing-hip sequences; assert they remain below candidate persistence. Assert the generated question rejects browsing, phone handling, clothing adjustment, open carrying, and trolley/basket placement.

- [ ] **Step 2: Run the tests and confirm RED**

Run: `./.venv/bin/python -m pytest tests/test_gate_evidence_quality.py tests/test_prompt_regression.py -q`

- [ ] **Step 3: Carry detector cues into the event and prompt**

Extend `concealment_to_events()` without changing the event schema. Update the versioned concealment question and regenerate the expected prompt fingerprint through the repository's prompt-regression workflow.

```python
extra = {
    "destination": assessment.destination,
    "components": assessment.components,
    "reasons": assessment.reasons,
    "limited": assessment.limited,
    "associated_bag": assessment.associated_bag,
}
```

The prompt must include this exact policy sentence: `Reject normal browsing, phone handling, clothing adjustment, openly carried goods, and placement into a trolley or shopping basket.`

- [ ] **Step 4: Verify evidence selection**

Add a behavioral assertion that a concealment rule receives three chronological full frames and an appended subject crop when a valid box exists.

- [ ] **Step 5: Run focused tests and commit**

Run: `./.venv/bin/python -m pytest tests/test_concealment.py tests/test_gate_evidence_quality.py tests/test_prompt_regression.py -q`

Commit: `feat(concealment): pass grounded temporal cues to verification`

---

### Task 4: Validate Scenario 10 End To End

**Files:**
- Modify: `docs/PROJECT_CONTEXT.md`
- Create: `docs/CHI_PILOT_TESTING.md`
- Test: existing concealment, serving, rules, gate, and pipeline suites.

**Interfaces:**
- Produces: repeatable positive/negative test matrix and exact run commands.

- [ ] **Step 1: Document the pilot matrix**

Record pocket-positive, bag-positive, trolley-safe, phone-to-pocket, clothing-adjustment, browsing, and open-carry cases with labeled event intervals and expected verdicts.

- [ ] **Step 2: Run the focused regression suite**

Run: `./.venv/bin/python -m pytest tests/test_concealment.py tests/test_heavy_models_earn_their_frames.py tests/test_serving.py tests/test_prompt_regression.py tests/test_gate_evidence_quality.py -q`

- [ ] **Step 3: Run the full Python suite**

Run: `./.venv/bin/python -m pytest`

- [ ] **Step 4: Commit documentation**

Commit: `docs(pilot): add Chi concealment acceptance matrix`
