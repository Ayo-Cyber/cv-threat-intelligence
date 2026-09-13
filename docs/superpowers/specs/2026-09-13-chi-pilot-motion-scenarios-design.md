# Chi Pilot Motion Scenarios Design

## Goal

Implement Chi pilot scenarios 4 and 5 without turning ordinary movement into alert noise:

- Scenario 4: visibly track people moving normally inside permitted areas.
- Scenario 5: detect a configurable number of people moving simultaneously.

## Product Behaviour

Scenario 4 is operational telemetry, not a threat alert. YOLO detects people, Supervision ByteTrack maintains their identities, and a new motion tracker classifies each track as stationary or moving. A moving person inside a permitted zone may be shown with a green box and a label containing the track ID, movement state, and zone. No candidate is sent to TrueSight merely because normal movement occurred.

Scenario 5 is a distinct event. It fires one aggregated `multiple_people_moving` candidate when at least `movement_min_people` distinct tracks are moving during the same sustained interval. It does not require those people to be clustered, and it does not reuse `CrowdFormationDetector`, because a crowd can be stationary while several moving people can be spread across a permitted area.

## Motion Model

Create `cvti/detector/person_motion.py` with two focused units:

- `PersonMotionTracker` consumes tracked person boxes and timestamps. It calculates center displacement per second, normalizes it by frame diagonal, smooths it with an exponential moving average, and uses separate enter/exit thresholds to prevent flicker. It retains a track through short detector dropouts and expires stale tracks.
- `SimultaneousMovementDetector` consumes the current `PersonMotion` snapshots. It requires a configurable person count and persistence period, latches after firing, and resets only after the moving count falls below threshold.

Movement defaults must be conservative and configurable per camera. The implementation records measured speed, duration, track IDs, zone names, and the group bounding box in candidate metadata.

## Zone Semantics

The existing `RetailZoneMonitor` remains the source of zone membership. Site configuration may declare `permitted_movement_zones`. When this list is present, scenario 4 overlays and scenario 5 counts include only tracks in those zones. When absent, the whole camera view is treated as permitted.

The latest entry, presence, dwell, and exit behaviour remains unchanged. Movement classification must not reset zone dwell state or create duplicate entry/exit alerts.

## Live Overlay Controls

Tracking continues internally regardless of overlay visibility.

The UI adds a global `Show tracking` toggle to both the standard camera wall and streams-only wall. Each camera tile adds a three-state override:

- `Use global`
- `Show`
- `Hide`

The global preference is persisted per operator in local UI storage. Camera overrides are session-only. The default is hidden.

The frame publisher serves both raw and annotated variants from the same decoded frame. A `tracking=1` stream query selects the annotated variant. Annotated frames are generated only while at least one tracking-overlay viewer is connected, avoiding permanent duplicate JPEG work. The smooth publisher passes its cached moving-track overlays so boxes remain fluid between detector samples.

Only moving people receive normal-operation overlays. Normal movement uses green; a scenario-5 candidate uses amber; an alert associated with a track uses the existing alert colour. Stream authentication rules remain identical for raw and annotated variants.

## Pipeline Integration

`PerCameraState.process()` updates motion immediately after ByteTrack. It stores current overlay records independently of alert generation. Scenario 5 creates a `RawEvent(detector="multiple_people_moving")`, which enters the existing Customization Engine, queue, Agent Mapper context, and TrueSight verification flow.

Add a rule to the Chi pilot ruleset rather than the always-on critical baseline. The gate receives three chronologically spread frames and asks whether multiple distinct people are visibly moving at the same time. Verification failure remains fail-visible under existing policy.

## Configuration

Per-camera keys:

- `normal_movement`: boolean, default `false`
- `multiple_people_moving`: boolean, default `false`
- `movement_enter_speed_ratio`: positive float
- `movement_exit_speed_ratio`: positive float lower than enter threshold
- `movement_min_track_seconds`: positive float
- `movement_min_people`: integer at least 2
- `movement_persistence_seconds`: positive float
- `permitted_movement_zones`: optional list of configured zone names

Invalid combinations fail site-config validation with a camera-specific message.

## Testing And Acceptance

Automated tests cover normalized speed, jitter rejection, hysteresis, stale-track expiry, simultaneous count, persistence, latching, zone filtering, configuration plumbing, raw versus annotated stream selection, overlay viewer lifecycle, global preference persistence, and camera override precedence.

Pilot acceptance uses labeled positive and negative video intervals. Record person-detection recall, ID switches, scenario-5 precision/recall, duplicate alerts per incident, detection delay, and throughput with overlays hidden and shown. Scenario 4 passes when moving people in permitted zones retain stable boxes without producing alerts. Scenario 5 passes when the configured simultaneous threshold fires once and stationary groups remain quiet.

## Non-Goals

- Crowd density or unsafe crowd formation; the existing detector remains separate.
- Biometric identity or face recognition.
- Inferring criminal intent from ordinary movement.
- Training a new model for the pilot.
