# CHI Pilot: 13 Required Scenarios

Source: CHI scenario-table screenshot provided by Demi.
Recorded: 2026-10-02. Wording below preserves the requirements, with minor
spelling and punctuation corrections.

| No. | Scenario | Assigned to Demi |
| --- | --- | --- |
| 1 | Person entering/exiting | |
| 2 | Vehicle entering/exiting | |
| 3 | Object identification (CHI products, arrangements, loading, trucking) | Yes |
| 4 | Normal movement within permitted areas | Yes |
| 5 | Multiple people moving simultaneously | Yes |
| 6 | Person entering a restricted zone | |
| 7 | Person remaining in an area beyond the configured time | |
| 8 | Unusual movement/activity | |
| 9 | Object left behind/removed | Yes |
| 10 | Pocketing or bagging product | Yes |
| 11 | Excessive crowd/occupancy | |
| 12 | PPE violation or unsafe behaviour detection where applicable | |
| 13 | Fire and smoke detection | |

## Scope and Priorities

Demi initially prioritised 4, 5 and 10, followed by 3 and 9. Person bounding
boxes should have a display toggle; hiding boxes must not disable detection
or tracking. Normal permitted movement is not itself an alarm condition.

Scenario 3 is broader than reference-photo matching: recognising a CHI product
does not by itself establish that loading, trucking or an arrangement is correct.
Agree those sub-scenarios and expected outputs explicitly with CHI.

## Readiness Assessment

**Not yet demonstrated ready for all 13 scenarios.** A supervised pilot with a
clearly agreed, validated subset is different from production readiness or a
claim that every requirement is met.

This document is not a fresh audit of Ayo's latest remote code. Earlier work and
passing automated tests do not establish current end-to-end performance on CHI
footage, hardware or a packaged release. No scenario is marked accepted here
without a recorded acceptance result.

Known evidence from our recent work:

- Independent camera preview was merged in PR #198, commit `84b3418`.
  Linux, frontend and RTSP-to-alert CI checks passed. Physical-camera handover
  and packaged Windows acceptance still need testing.
- The Windows CI run after that merge had the same 21 failures as the preceding
  main run. These concern file handles, backup collisions and platform-specific
  assumptions; they must not be presented as a clean Windows result.
- The inspected object-watch implementation requires local SigLIP artifacts.
  Its automatic customer download was proposed in
  `AYO_SIGLIP_MODEL_DOWNLOAD_HANDOFF.md`, not implemented by that document.
  Confirm any subsequent work by Ayo before treating this as a current blocker.

## Acceptance Checklist

All rows below remain **unverified in this checklist**, not necessarily
unimplemented. Test the exact release intended for the pilot.

| No. | Minimum evidence to record |
| --- | --- |
| 1 | Entry and exit direction/counts are correct; lingering at the boundary does not repeatedly count the same person. |
| 2 | Vehicle entry and exit work with the configured boundary, including closely spaced vehicles and stationary vehicles near it. |
| 3 | Enrolled CHI products match across representative views; similar non-target products are rejected. Test arrangements, loading and trucking as separately agreed requirements. |
| 4 | Moving people remain visible/tracked in permitted areas without inappropriate alarms; box display toggles independently of inference. |
| 5 | Multiple simultaneous people are detected, with tracking checked through crossings and partial occlusion; measure misses and identity switches. |
| 6 | Actual entry into the drawn restricted zone triggers the configured event; people outside it do not. Verify anchor/overlap behaviour. |
| 7 | Dwell timing is correct at, below and above the configured threshold, including leaving and re-entering the zone. |
| 8 | Define specific unusual behaviours and normal counterexamples with CHI before judging accuracy; avoid an undefined catch-all anomaly promise. |
| 9 | Test an object being left behind and an object being removed separately, including occlusion, camera movement and permitted object handling. |
| 10 | Test pocketing/bagging sequences and benign lookalikes such as adjusting clothing or handling personal belongings; track both missed events and false alerts. |
| 11 | Occupancy thresholds, persistence and clearing behaviour work; normal flow and temporary overlap do not repeatedly alarm. |
| 12 | Agree applicable PPE and unsafe behaviours, supported detectors and camera visibility. Test compliant and non-compliant examples. |
| 13 | Use safe prerecorded fire/smoke footage and confusing negatives such as steam, glare and dust. Verify alert delivery, not just detector output. |

## Pilot Go/No-Go

1. Freeze and record the exact commit/build, OS, camera setup, models and configuration.
2. Agree must-pass scenarios, numeric acceptance thresholds and unsupported cases with CHI.
3. Test representative positive and negative footage, then the actual site cameras.
4. Record expected versus actual output, missed events, false alerts, duplicates,
   latency and saved evidence for each scenario. Do not use mock verification to
   claim real recognition or alert-verification performance.
5. Verify clean installation/model download, camera preview, monitoring start/stop,
   scene review, zone persistence, reconnects and notification delivery end to end.
6. Run for a representative operating period on the intended machine and camera
   count, checking resource use and whether feeds and alerts remain responsive.
7. Proceed only with the validated scope, named operator oversight and documented
   limitations. Do not treat this pilot as a replacement for established safety systems.

Suggested result record: scenario, build/config, clip/camera, expected outcome,
actual outcome, latency, evidence link, pass/fail, owner and outstanding issue.
