# CHI Validation Checkpoints - 2026-10-03

## Follow-up 2026-10-04: structured observations without extra inference

Real desktop run `evidence/2026-10-04T10-13-44.097Z/` under scenario10:
2 verified, both rejected by the new temporal consistency check, no errors or
pending verdicts at snapshot. Raw VLM claimed pocket insertion but start/end
indices were both3. Do not count rejections as validated negatives: this clip is
user-labelled concealment and reliable recall remains unresolved.
Median verification19.9s (prior run24.12s); processed359 frames, ingest24fps,
no stale drops or reconnects. No added model calls/images/output budget or stream
configuration changes. Small-run evidence only, not a no-regression guarantee.
Descriptions now derive from validated categorical observations, not unrestricted
model prose. The fields themselves can still hallucinate; held-out evaluation
and the missing retail action checkpoint remain outstanding.

## Follow-up 2026-10-04: timed camera notices

Actual Electron/local engine/Ollama run saved under
`runs/chi_validation/scenarios/10_concealment/evidence/2026-10-04T08-53-47.219Z/`.
Inspected notice-31.png (amber unverified gesture) and notice-55.png (later review
required). Incident list shows the new "Possible product concealment" title.
2 gate confirmations, 0 errors, 0 pending, median24.12s at saved snapshot.
No claim of ground-truth accuracy: new assessment still mislabels destination as
pocket despite the prompt instructing waistband/clothing distinction. Existing
incident history is retained, not rewritten. Notice expiry and camera scoping
covered by Python/TS and browser checks; production frontend build passes.

Status: preliminary diagnostics, NOT acceptance of all five KPIs.
Checkout: `feat/chi-object-state-events`, base `adf0e6b`, plus local KPI 9 changes.

## Latest KPI10 checkpoint: temporal fixes and structured verification

Final visible Electron/local Ollama single-wig run, after temporal scoring,
rewind, cooldown and evidence/prompt fixes:
`runs/chi_validation/scenarios/10_concealment/evidence/2026-10-03T21-38-27.341Z/`.
Saved health reports 3 verified, 2 confirmed, 1 rejected, 0 errors, 0 pending,
2 deduped and 22.37s median gate latency. Screenshots and gate inputs/outputs
are archived there. Shared event database and cumulative monitor.log contain
older results; do not count them as new detections.

Alert delivery on the wig clip improved, but these are not validated true
positives: one confirmation misidentifies the object as a mannequin. Isolated
verification of the other retail clip still risks confusing different actors.
Structured insertion fields constrain output but do not establish visual truth.
KPI10 remains under validation, not accepted. 111 focused regression tests passed.
See CONCEALMENT_RELIABILITY_HANDOFF.md for reproduction and next steps.

## KPI10 corrected-crop rerun

Actual Electron/local engine/Ollama 120s run with test-only 8fps/stride1:
`runs/chi_validation/scenarios/10_concealment/evidence/2026-10-03T21-14-38.415Z/`.
5 verified, 3 confirmed, 2 rejected, 0 errors, 211 deduped, 12 pending at saved
health snapshot. Median gate latency18.65s. These are partial-run outcomes, not
a fully drained evaluation. Memory guard reduced requested8fps to4.
All new confirmations on retail_2; explanations say touching waist, grasping a
black bag, or holding an object against waist. They do not establish product
insertion. Therefore these are unsupported confirmations requiring review, NOT
successful concealment detection. No confirmed retail_1 alert. Original misses
remain unresolved. Cumulative monitor.log contains earlier runs too.

Gate artifacts now contain subject_bbox from pose. Inspected gate_0002 crop
shows the black-clothed walker; the new crop path is active. Added two regression
tests verifying the pose box wins over a colliding ByteTrack ID and that missing
pose box does not select an unrelated person. Both passed, supplementing earlier
61 tests. Production sampling/thresholds unchanged. Reproduce with
`node scripts/chi-kpi10.mjs --dense` from Frontend after preparing inputs.

## KPI10 miss trace follow-up

User reports actual concealment in both clips; no-alert results are suspected
misses, not successful negative controls. Original gate artifacts copied to
`runs/chi_validation/scenarios/10_concealment/trace/original-gate/` before reruns.
Inspected gate_0001/frame_0 and gate_0002/frame_3: early walking interval, not
evidence that later action was checked. First candidate timestamp is 3.0697s.
Dense half-second source sheets: retail_1 8-14s and 14-20s, retail_2 4-10s.
Occlusion remains substantial; frame sheets do not establish complete event truth.

Offline comparison, same model/512 input/.4 confidence/.63 concealment threshold,
8fps requested and heavy_stride=1:
- retail_1: 233 samples, 163 pose updates, 217 poses, max score .8653,
  30 candidate assessments; first at 18.133s, plus later walking candidates.
- retail_2: 79 samples, 75 pose updates, 133 poses, max score .7133,
  two candidates at 3.604 and 3.737s, still before the later interaction.
Report: `pose-probe-8fps.json`. This is a sensitivity experiment, NOT a production
setting change, accuracy improvement claim, or VLM-confirmed success. More pose
sampling costs compute and can increase nuisance candidates. Dwell and persistence
currently count frames, so cadence changes behavior, not just latency.

Code inspection also found pose IDs were looked up in ByteTrack's independent ID
space when choosing a verification crop. Fixed by propagating subject_bbox from
PoseFrame -> ConcealmentAssessment -> event metadata -> serving crop. No unrelated
largest-person fallback for concealment; missing bbox retains full-frame evidence.
61 concealment/entrypoint/serving tests pass. End-to-end rerun after fix outstanding.

## KPI10: first real desktop concealment run

Evidence: `runs/chi_validation/scenarios/10_concealment/evidence/2026-10-03T21-01-44.331Z/`.
Prepared by `scripts/prepare_kpi10_test.py`; visible desktop runner
`Frontend/scripts/chi-kpi10.mjs`, 120-second test. Sources: existing local
`data/test_clips/theft_shop_01.mp4` and `theft_shop_02.mp4`. Contact sheets saved
in scenario folder. Filenames/news captions are NOT ground truth. These are
not clean independently labelled positive/negative controls.

Serving: YOLO nano + shared yolov8n-pose, 512 input, .4 object confidence,
target 4fps, default heavy_stride=2. Actual throughput/sampling differs from
target. Concealment thresholds unchanged. Dedicated product_concealment rule
requires visible pocketing/bagging and rejects inference from captions.

Health snapshot: 364 processed frames per camera, 2 verified, 0 confirmed,
2 rejected, 7 deduped, 0 pending, 0 gate errors. Ollama gemma3:4b median
latency 18.69s; memory healthy at 2.93GB available. Both rejections were on
retail_2 and described walking with a hand near the waist. This proves the
candidate-to-VLM path ran, not positive concealment recall or correct rejection
of every possible action. Old KPI5 events remain visible in the shared test DB.
One inspected screenshot shows retail_2 temporarily CONNECTING; do not claim
uninterrupted preview. Screenshots, config, screen text and health are saved.

Separate real-model diagnostic (`scripts/check_kpi10.py`, `pose-probe.json`):
retail_1: 117 samples, 47 pose updates, 63 poses (60 complete), max score .58754,
zero candidates. retail_2: 45 samples, 23 pose updates, 39 poses (36 complete),
max score .76302 but no sustained candidate. Complete means all returned
required keypoints non-null, not validated keypoint accuracy. Offline source
timestamps and fixed sampling differ from live wall-clock sampling and loops;
the lack of offline candidates does not contradict the live candidates.

23 concealment and entrypoint unit tests passed. KPI10 remains unvalidated:
need independently reviewed positive intervals, benign handling controls, and
sampling/pose diagnostics. Do not lower thresholds just to manufacture alerts.

## Additional real-video checkpoint: Meet_Crowd

Downloaded real acted footage and manual XML annotations from the EC Funded
CAVIAR project/IST 2001 37540 (CC BY-SA):
https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/
Files: `runs/chi_validation/caviar/Meet_Crowd.mpg` and `mc1gt.xml`.
Also inspected `ms3ggt.xml` and `mws1gt.xml`. These meeting clips do not supply
a sustained annotated stationary-group interval. Do not count them as negative
passes. Meet_Crowd is a new positive movement check, NOT completion of the
requested natural stationary-group negative test.

Actual Electron + local engine + Ollama gemma3:4b run:
`runs/chi_validation/scenarios/05_multiple_people_moving/evidence/real-crowd/2026-10-03T20-31-18.768Z/`.
`monitoring-2.png` shows the new retail_walk alert (camera ID reused by harness;
its source in this run is Meet_Crowd). One verified/confirmed alert, confidence
0.95, zero gate errors, two deduplications, zero pending at health snapshot.
Earlier crowd_walk alert remains visible and is not a new result.
VLM latency 43.92 seconds; total alert latency 43.97 seconds.
Memory guard reduced detector input 960 -> 320; health degraded with 1.83GB
available. This run does not establish consistent high-resolution performance.

Sequential fixed-resolution model probe: `real-crowd-probe.json` in scenario
directory. Meet_Crowd: 41 samples, 20 with tracks, 13 with movers, maximum four
tracks/four movers, candidate at 9.12s. Walk1 control: 51 samples, max one
track/mover, no candidate. Sampling metrics are not recall or mAP.
No production thresholds changed. Existing CAVIAR calibration was reused.

Replay from repo root:
```sh
python3 scripts/prepare_kpi5_real_crowd.py
cd Frontend
node scripts/chi-three-videos.mjs --four --kpi5 --real-crowd --demo
```
Assets remain local/ignored. The finite run omits `--demo`, records screenshots,
and stops monitoring. Natural standing-group footage and jitter negatives
remain outstanding; do not advance this to universal KPI5 acceptance.

## KPI5 negative controls

Desktop run: 436 total frames across four cameras, zero queued alerts and zero
context suppressions. No VLM calls were needed. The single alert visible in
screenshots belongs to the earlier positive run; its timestamp predates this run.
No events were deleted or relabelled to make the negative test look clean.

| Offline real-model control | Samples | Frames with tracks | Frames with movers | Max tracks / movers | Candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| CAVIAR Walk1, 24.48s | 51 | 21 | 15 | 1 / 1 | 0 |
| Frozen crowd image, 30s | 63 | 63 | 0 | 38 / 0 | 0 |

Frozen group is explicitly synthetic: first frame of normal_street_01 repeated
using cv2.VideoWriter, not evidence of natural stationary-group reliability.
This checks multiple visible detections alone do not imply group movement.
Single-person negative is not entirely explained by missed detections: actual
moving tracks were measured in 15 samples. These are not recall measurements.

Actual Electron screenshots/site-config/gate-health/monitor log:
`runs/chi_validation/scenarios/05_multiple_people_moving/evidence/negative-controls/2026-10-03T20-14-23.679Z/`.
Detailed counts: `runs/chi_validation/scenarios/05_multiple_people_moving/negative-probe.json`.
Runtime retained the four-camera layout; street_two reused for Walk1, retail_walk
for the frozen control. The other two cameras had KPI5 rules disabled. No
production camera configuration changed. scripts/check_kpi5_negatives.py runs
the separate sequential probe, not the UI or gate.

Next validation: natural standing groups with body sway, one mover amongst
stationary people, camera motion, occlusion/reappearance, and permitted zones.
KPI5 has a confirmed positive and these negative controls, not full acceptance.

## KPI 5: first positive desktop checkpoint

Enabled multiple_people_moving and configs/chi_pilot_v1.json only for crowd_walk
in runs/chi_validation/scenarios/05_multiple_people_moving/site.json. Other
three feeds retained KPI4 observation settings. Real Electron + Python engine +
Ollama gemma3:4b produced a confirmed chi_multiple_people_moving event at 0.90.
Gate reason: numerous people visibly moving simultaneously. Event evidence:
`runs/chi_validation/desktop/events/20261003_210459_crowd_walk_chi_multiple_people_moving`.
Desktop screenshots, saved config, result and monitor log:
`runs/chi_validation/scenarios/05_multiple_people_moving/evidence/four-videos/2026-10-03T20-03-38.303Z/`.
31 person-motion unit tests pass, including one mover and stationary-group
negative checks. This does NOT replace negative video tests, which are next.
No full KPI5 acceptance yet. High-priority event comes from the test rule;
ordinary simultaneous movement is not inherently a security threat.

## Latest: exact-frame crowd box display

Issue: smooth video continued while boxes represented an older sampled image.
At 2fps plus shared inference latency, this produced visibly trailing boxes.
No evidence that CAVIAR's rotation was applied to crowd_walk; calibration is
camera-specific. Added opt-in `box_display: "synchronized"` to crowd_walk.
Tracking viewers receive exact inference images with corresponding boxes;
smooth loop cannot overwrite them with newer frames. Boxes-off returns to
smooth viewing. Existing raw/tracking viewer accounting is reused.

Actual Electron evidence: evidence/four-videos/2026-10-03T19-45-34.601Z under
runs/chi_validation/scenarios/04_normal_movement. monitoring-1.png inspected:
crowd boxes align with subjects in the displayed sample. No full-video accuracy
claim. 70 existing serving/publisher regressions and 3 new exact-frame tests
passed. New tests cover camera/viewer isolation, inference source-image identity,
blocking newer smooth frames and returning to smooth mode after boxes-off.

Tradeoffs: ~2fps annotated playback or lower under load, plus processing delay.
Raw viewers of that camera also get sampled cadence while any tracking viewer
is active. Other cameras unchanged. No optical-flow extrapolation or invented
positions; localization noise and missed detections still possible. Raising
sampling/model capacity costs more compute; exact-frame mode preserves spatial
alignment without claiming higher detector accuracy. Screenshot cadence is not
an FPS benchmark. Coworker command saved in top-level DEMO_FOUR_CAMERAS.md.

## Latest: CAVIAR orientation/model correction

Publisher confirms wide-angle lens and 384x288 resolution:
https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/
Downloaded mwt1gt.xml from its linked Meet_WalkTogether1/ directory and
official YOLOv8s weights from Ultralytics assets (v8.3.0 release).
Rotation-only probes sometimes increased furniture false detections. A larger
model alone did not solve the recall problem either. Selected camera-specific
YOLOv8s + -60 degree inference rotation + 0.5 confidence, keeping 960px input.
This corrects orientation for this view, not wide-angle distortion generally.
No exclusion masks or hardcoded furniture positions were used.

| Diagnostic sample | Before TP / FP / FN | After TP / FP / FN |
| --- | --- | --- |
| Frames 0,50,...700 | 3 / 3 / 16 | 13 / 0 / 6 |
| Frames 25,75,...675 | 1 / 4 / 16 | 11 / 0 / 6 |

Before = nano, original orientation, conf0.25. After = small, -60, conf0.5.
Metrics use IoU >=0.3 and greedy one-to-one matching against published boxes;
they are diagnostic and NOT standard COCO mAP, full-video recall or independent
validation. Some stationary people are unlabelled in CAVIAR. Parameters were
selected using first sample; offset25 is an interleaved same-video check.
Reports/annotated predictions: runs/chi_validation/caviar_models/results.json,
metrics.json and offset_25/ (script now writes offset_0/ for a fresh first run).

Live desktop evidence:
`runs/chi_validation/scenarios/04_normal_movement/evidence/four-videos/`
`2026-10-03T19-24-57.664Z/`.
monitoring-0.png shows two real people boxed in CAVIAR, not the plant/bench.
monitoring-1.png shows empty CAVIAR without furniture boxes. Actual log reports
raw_people=2, tracked=2, overlays=2 at one checkpoint. Other feeds and both
overlay toggle scopes remain operational. Nighttime configuration was not
modified; existing intermittent detection remains. No claim of perfect recall.

Implementation: optional per-camera detection_inference validated/cached by
CameraInference; shared default inference unchanged. Rotated boxes inverse-map
all four corners and clip to original dimensions before downstream processing.
Local weights required; matching class maps enforced. Source video/zone geometry
is never rotated. Extra model costs memory/compute; observed batch timings around
420-485ms in this run (offline diagnostics also used CPU, so not a clean FPS
benchmark). No weights bundled, no production site edited, no commit/push.
Final focused regression suite: 123 passed (camera inference, desktop settings,
serving, frame publisher and person motion). Includes inverse-coordinate mapping,
per-camera isolation, invalid settings and tensor/numpy outputs.

Reproduce visible four-camera test from the repo:
`cd Frontend` then `node scripts/chi-three-videos.mjs --four`.
This uses isolated test accounts/configuration under runs/chi_validation, starts
real monitoring, takes screenshots, verifies switches and stops/closes afterward.
Keep the downloaded small-model weights at their configured local path.

## Latest checkpoint: four-camera desktop coverage

Tested the actual Electron app and Python engine with four concurrent feeds:
normal_street_02, theft_shop_01, CAVIAR Meet_WalkTogether1, normal_street_01.
The last is the earlier crowd video requested by the user.

Diagnosis: default desktop inference is 512px, confidence 0.4. Same-frame
offline comparisons at 10 seconds and confidence 0.25 yielded crowd detections
25 -> 31 at 512 -> 960, retail 1 -> 2, CAVIAR 0 -> 1; night remained 0.
Separately, tracker association dropped some detected people from the display.

Changes: optional validated site inference settings; diagnostic opt-in
ARGUS_TRACKING_DIAGNOSTICS=1; neutral PERSON overlays for unmatched detections
when normal movement is enabled without explicit permitted-zone restrictions.
These overlays do not become identities, movement counts or alert candidates.
The publisher excludes their transient identifiers from track metadata.

Final run used imgsz=960, confidence=0.25, target_fps=2, CPU, existing YOLOv8n,
and Ollama configured for verification. This is a box-display test, not proof
of VLM verification or alert correctness. Default production settings remain
512/0.4/4. Lowering confidence affects detector candidates generally and may
increase false positives; the test setting is not a production recommendation.

Evidence: `runs/chi_validation/scenarios/04_normal_movement/evidence/`
`four-videos/2026-10-03T18-57-56.508Z/`. Contains three monitoring screenshots,
single-camera-off, all-off, narrow-layout, result.json, site-config.json and
monitor.log (append history includes prior runs). The final run is the last
serving startup in the log. Both toggle scopes passed in the desktop app.

Result: PARTIAL. Crowd boxes improved across the image; one runtime sample had
40 raw detections, 32 tracks and 40 overlays. Retail has visible person boxes.
Nighttime and CAVIAR detection remains intermittent/poor. A likely false
CAVIAR box appears in monitoring-2.png. Some moving boxes lag the smooth feed;
check overlay/frame timing before acceptance. Runtime inference averaged about
300ms per four-frame batch, versus about 123ms in the earlier 512px run;
these are compute timings, not achieved display FPS. No ground-truth recall
or precision measured yet. All 112 focused regression tests passed.

Next: evaluate harder clips with labelled frames, compare a stronger detector
or region-based inference, and verify temporal alignment. Keep KPI 4 open;
do not advance to KPI 5 on the strength of the crowd screenshot alone.

## Sources and isolation

CAVIAR Walk1, Meet_WalkTogether1 and LeftBag_PickedUp were downloaded from
https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/ .
The publisher identifies Creative Commons BY-SA usability and requests credit
to the EC-funded CAVIAR project / IST 2001 37540. Keep that attribution if
redistributing results/footage. Approximately 30 MB downloaded, plus Walk1 XML.
These are 384x288, 25-fps research clips, not representative CHI camera footage.
Existing local theft_shop_01 and theft_shop_02 were also probed. Their filenames
are not substitutes for independently labelled concealment event intervals.

Test configuration, downloaded videos, reports, local accounts and screenshots
are under ignored `runs/chi_validation/`. Existing site configuration was not edited.
No webcam used; no frames sent to a cloud VLM. Ollama `gemma3:4b` was local.

## Checks completed

- KPI 9 synthetic/serving tests: 27 passed.
- Concealment entrypoints and object-watch runtime/integration/embedding tests:
  46 passed. These are engineering tests, not pretrained recognition accuracy.
- Frontend production build: passed.
- Tracking overlay, whole-view zone and object-watchlist unit tests: 13 passed.
- Electron launched the actual app and local API in a separate profile, rendered
  the account screen, and captured no renderer errors. See desktop/startup.png.
  This initial check does not prove monitoring, zone drawing or alerts in the UI.
- Follow-up Electron check: disposable owner registration succeeded; backend
  connected; onboarding reported on-device AI ready. No renderer errors.
  The latest startup screenshot records this authenticated onboarding screen.
- npm reported 14 high and 1 critical dependency vulnerabilities. Not remediated
  in this testing task; do not treat build success as a security clearance.

## Real pipeline run

Walk1 was run with YOLO, pose, Agent Mapper and Ollama configured, target 4fps,
512px inference, 45-second run budget. Processed 113 frames; published 198;
zero candidates, zero VLM alert verifications. This proves startup/inference/
mapping execution, NOT alert-verification success. Mapping stayed unreviewed.
The mapper reported estate_gate / estate_street at confidence 0.7 for the
INRIA lobby view. Treat this as an incorrect/unsupported environment proposal
requiring review, not trusted site context.
Memory guard trimmed buffers once; available memory around 2 GB, swap 26.5 GB.

## Offline whole-clip detector probe

`scripts/chi_video_probe.py` runs real YOLO and production PerCameraState with
pose, movement and concealment enabled, at approximately 4 samples/sec, CPU,
confidence 0.4 and image size 512. It deliberately does not invoke the mapper
or gate. It is diagnostic and has no reviewed zones or product enrollments.

| Clip | Samples | Raw-person frames | Tracked-person frames | Moving-overlay frames | Maximum tracked people |
| --- | ---: | ---: | ---: | ---: | ---: |
| Walk1 | 102 | 5 | 3 | 0 | 1 |
| Meet_WalkTogether1 | 118 | 2 | 0 | 0 | 0 |
| LeftBag_PickedUp | 226 | 16 | 8 | 1 | 1 |
| theft_shop_01 | 117 | 83 | 79 | 44 | 2 |
| theft_shop_02 | 45 | 43 | 43 | 29 | 3 |

All five yielded zero queued candidates. None yielded a supported bag detection.
Counts are not recall: XML/event ground truth has not yet been scored against
predictions. The CAVIAR probe reveals poor raw detection with this configuration,
so no successful group-motion or bag-removal conclusion can be drawn.
KPI 9 was not enabled in this generic probe: it needs a reviewed position zone;
absent bag detections already block the intended bag-based end-to-end test.
Retail movement overlays do work, but concealment recall remains unproven.

## Remaining / blockers

- KPI 3: SigLIP weights were not found in the checked local model/library/cache
  locations. Need a provisioned model and enrolled reference images before a
  real matching test; no CHI recognition success claimed.
- KPIs 4/5: investigate small/angled CAVIAR subjects, compare resolution/model/
  thresholds against annotated ground truth, then validate permitted zones.
- KPI 9: supported object detections, reviewed single-position zones, positive
  placement/removal and negative occlusion sequences, then real gate evaluation.
- KPI 10: independently label pocketing/bagging intervals and benign examples;
  inspect pose coverage and candidate generation before tuning thresholds.
- Finish desktop preview, monitoring, zone editing, overlay switching and alert
  review with the same configuration. Current startup smoke is not that test.
- Real site footage, CHI-specific products and all required acceptance thresholds
  still need agreement. Do not declare the pilot ready from these results.
