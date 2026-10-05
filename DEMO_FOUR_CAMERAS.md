# Four-camera coworker demo

Run this in Terminal on this Mac:

```sh
cd "/Users/macbook/Desktop/CV Threat Intelligence/argus-camera-preview/Frontend"
node scripts/chi-three-videos.mjs --four --demo
```

This opens the real Electron app, signs into the isolated validation account,
enables person boxes and starts the local Python engine. It stays open for
your presentation. Click Stop monitoring before closing the window.
Do not run another copy on port 8799 at the same time.

Feeds: nighttime street, retail, CAVIAR pair and the earlier crowd video.
Use Person boxes on each camera, or Person boxes: all cameras.

The crowd camera now uses exact-frame annotations: boxes and people belong
to the same sampled frame. Playback with boxes is consequently about 2 updates
per second (possibly lower under load), not smooth full-rate video. Turn boxes
off to return to smooth playback. While any viewer requests boxes for that
camera, its raw viewers also receive the sampled cadence. Other cameras are
not affected. This prevents timing drift, not all detector misses or loose boxes.

CAVIAR uses the downloaded YOLOv8s model and -60-degree inference orientation
correction. Its displayed image is unchanged. This improves the tested clip;
it is not a universal fisheye calibration or a promise of perfect detection.

Local prerequisites already installed for this test: Frontend dependencies and
production build, the Python environment referenced in the runner, test account,
four source clips, and runs/chi_validation/yolov8s.pt. These assets are ignored
test data, not bundled with a clean clone. The script uses port 8799 and the
isolated runs/chi_validation/desktop profile, not your production account.

For the automated screenshot/check run, which closes after testing:

```sh
node scripts/chi-three-videos.mjs --four
```

Evidence is saved under runs/chi_validation/scenarios/04_normal_movement/evidence/four-videos/.

## KPI 5 validation commands

From the same Frontend directory:

```sh
node scripts/chi-three-videos.mjs --four --kpi5
node scripts/chi-three-videos.mjs --four --kpi5 --negative
```

Run separately, waiting for the first to finish. Positive enables simultaneous
movement on crowd_walk. Negative uses street_two for CAVIAR Walk1 and retail_walk
for an explicitly frozen crowd-image control, not natural standing people.
Both run the real app and close after evidence capture. Add --demo to keep open.
The isolated database retains prior events, so an old positive alert can remain
visible during the negative run. Compare timestamps and saved gate-health.json,
not the total sidebar count, when judging whether new alerts fired.
