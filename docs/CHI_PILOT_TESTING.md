# Chi Pilot Testing

## Scope And Current Status

This document covers Chi pilot scenarios 4, 5, and 10. Scenario 4 is
non-alert telemetry: it draws boxes only around people classified as moving in
permitted zones. Scenario 5 creates one latched aggregate candidate for a
sustained interval with the configured number of simultaneous movers. It does
not require a crowd, proximity, clustering, panic, or unsafe behavior.

The motion implementation and focused automated suites pass. The current
prompt fingerprint includes the scenario-5 gate wording, while its precision
and recall remain unmeasured because the frozen corpus is unavailable. The
full Python regression is not green in this validation environment only
because local Ultralytics 8.4.64 differs from required 8.4.35. The local
redistributable clips also do not cover every controlled acceptance case.
Unmeasured cells below are deliberately not inferred from unrelated footage.

The repaired architecture has automated regression coverage, but the changed
TrueSight prompt is **unmeasured**. The frozen golden corpus is unavailable, so
[the prompt baseline](prompt_baseline.json) records zero scored cases and no
precision or recall.
The matrix below is the required recording and acceptance protocol, not a claim
that the seven Chi cases have already passed.

## Scenarios 4 And 5 Product Contract

YOLO person detections feed ByteTrack and the motion classifier whether or not
boxes are visible. Hiding or showing tracking changes only the stream variant
requested by the UI; it does not start, stop, enable, disable, or otherwise
change detection. The raw stream remains canonical.

Scenario 4 emits no candidate and no alert. A moving person receives a green
tracking box only when `normal_movement=true` and the track is inside one of
`permitted_movement_zones`. If that optional list is absent, the whole camera
view is permitted. Stationary people and moving people outside configured
permitted zones remain unboxed.

Scenario 5 is a separate detector. When at least `movement_min_people` permitted
tracks remain moving for `movement_persistence_seconds`, it emits one
`multiple_people_moving` aggregate candidate containing all qualifying track
IDs and their group box. The detector remains latched while the count stays at
or above threshold and resets only after the count drops below threshold.
Existing queue deduplication remains a second line of defense, but a queue
drop is not evidence that the latch emitted exactly once.

### Controls And Defaults

| Scope | Control | Default and lifetime |
| --- | --- | --- |
| Camera config | `normal_movement` | `false` |
| Camera config | `multiple_people_moving` | `false` |
| Camera config | `movement_enter_speed_ratio` | `0.05` frame diagonals/second |
| Camera config | `movement_exit_speed_ratio` | `0.02`; must be lower than enter |
| Camera config | `movement_min_track_seconds` | `0.4` seconds |
| Camera config | `movement_min_people` | `2` |
| Camera config | `movement_persistence_seconds` | `0.5` seconds |
| Camera config | `permitted_movement_zones` | absent, meaning the whole view |
| Operator wall | Global **Show tracking** | hidden; persisted per signed-in operator |
| Camera tile | **Use global**, **Show**, **Hide** | `Use global`; session-only override |

Changing the global control remounts only visible streams whose resolved
tracking value changes. Changing one camera override remounts only that camera.
Monitoring continues during both operations.

## Motion Recording Matrix

Record fixed-camera clips without cuts or digital camera motion. Fix labels
before the run and never move an interval after seeing a candidate or verdict.
Use the same camera, resolution, target FPS, model files, package lock, and
motion thresholds for the complete matrix.

| ID | Required recording and labels | Scenario 4 expectation | Scenario 5 expectation |
| --- | --- | --- | --- |
| `S4-P01` | **One-person permitted movement:** one person stands for at least 2s, then walks for at least 4s wholly inside `movement_permitted`. Label stationary and moving intervals. | One stable moving box only during the permitted moving interval; no alert. | No candidate because only one person moves. |
| `S4-N01` | **Stationary jitter:** one or more people remain stationary for at least 8s with ordinary detector/pose jitter. | No moving box; no alert. | No candidate. |
| `S5-P01` | **Two simultaneous movers:** two people begin walking, overlap in motion for at least 3s, then both stop. Label the exact simultaneous interval. | Both moving people boxed when permitted; no scenario-4 alert. | Exactly one candidate inside the simultaneous interval and one resulting alert at most. |
| `S5-N01` | At least three people stand close together for at least 8s. | No moving boxes. | No candidate: crowd proximity alone is irrelevant. |
| `S5-P02` | Two people walk simultaneously for at least 3s on opposite sides of the view. | Both permitted movers boxed. | Exactly one candidate despite spatial separation. |
| `S5-P03` | Two people move simultaneously; one is occluded for less than 1s and resumes the same path. Label mover visibility and the positive interval. | Visible moving tracks retain stable IDs where observable. | One incident and at most one shown alert; record any post-occlusion candidate as a duplicate candidate. |
| `S4-P02` | One person starts outside `movement_permitted`, crosses its polygon boundary, remains inside for at least 3s, then exits. Label outside, crossing, and inside intervals. | Box absent outside and present only while the moving track is inside. No alert. | That track counts only while inside; use a second permitted mover if validating the scenario-5 threshold at the boundary. |

The repository clips are not substitutes for this controlled matrix.
`normal_street_01.mp4` visibly contains multiple spread-out movers and supports
an exploratory scenario-5 replay. It does not provide a controlled stationary
crowd, temporary occlusion, or authored zone-boundary case. The other retained
clips do not complete those missing cells either.

## Motion Annotation Contract

Keep clips outside Git unless redistribution rights are documented. Retain a
SHA-256 manifest separately from the scoring labels. Combine every case in one
`labels.csv` with this exact header:

```csv
case_id,clip_sha256,clip_duration_s,target_fps,start_s,end_s,label,person_ref,expected_zone,scenario4_expected,scenario5_expected,incident_id
S4-P01,<sha256>,5.800,5,0.000,2.000,stationary,p01,movement_permitted,not_moving,negative,s4p01-stationary
S4-P01,<sha256>,5.800,5,2.000,5.800,moving,p01,movement_permitted,moving,negative,s4p01-moving
S5-P01,<sha256>,6.800,5,2.500,5.500,simultaneous_movement,p01|p02,movement_permitted,moving,positive,s5p01-move-1
```

`label` is an operator description; the scorer uses the separately named
expectations. `scenario4_expected` must be `moving`, `not_moving`, or
`not_scored`. `scenario5_expected` must be `positive`, `negative`, or
`not_scored`. Do not use a generic `expected` or `negative` field for a
scenario-4 positive. Rows sharing an `incident_id` must have identical bounds
and scenario-5 expectation. `person_ref` is the annotator's stable identity,
not a model track ID; join multiple people with `|` only when they share an
incident row. `clip_sha256`, `clip_duration_s`, and `target_fps` must be
identical on every row for one case. Here `clip_duration_s` means the timestamp
of the final decoded frame, not the container's nominal duration.

Times use the decoder timeline in seconds and intervals are half-open:
`[start_s,end_s)`. A sample exactly on a shared boundary belongs to the later
interval. Expected observation times use the clip-anchored grid
`timestamp_s = n / target_fps` and include every grid timestamp inside the
interval. For example, the first 5 FPS sample inside `[2.500s,5.500s)` is
`2.600s`. The final interval also requires the exact decoded frame at
`timestamp_s == clip_duration_s`. A row may differ from its expected timestamp
by at most `min(0.001s, 0.25 / target_fps)`; the 1ms ceiling matches the
documented timestamp precision, while the quarter-period ceiling prevents one
row from matching adjacent samples at high rates.

Combine frame-level observations for every case in `observations.csv`:

```csv
case_id,timestamp_s,person_ref,visible,detected,track_id,moving,zone,boxed
S5-P01,2.600,p01,true,true,17,true,movement_permitted,true
S5-P01,2.600,p02,true,true,23,true,movement_permitted,true
```

Sample at the configured `--target-fps`. A person-frame is scoreable only
when `visible=true`; use `visible=false` for the occluded interval rather
than counting an invisible person as a missed detection. Have a second person
review the labels before running acceptance. Use an empty `track_id` when
`detected=false`. Record `boxed` from the shown tracking capture; scenario 4
expects it only for moving people in a permitted zone. Write exactly one row
for every expected timestamp and every `person_ref` in its interval. A visible
miss is a row with `visible=true,detected=false`; never omit it. Missing,
duplicate, off-grid, or unexpected-person rows invalidate scoring so recall and
ID switches cannot improve through missing data.

Capture every generated scenario-5 candidate before queue admission and append
the eventual gate result to the same row. The scorer accepts either an exported
JSON document with a top-level `rows` list or a SQLite database containing a
`motion_candidate_audit` table. Both use exactly these fields:

```json
{
  "rows": [
    {
      "candidate_id": "S5-P01-0001",
      "case_id": "S5-P01",
      "timestamp_s": 3.100,
      "admission_status": "admitted",
      "gate_status": "confirmed",
      "persisted_event_id": "41"
    }
  ]
}
```

Allowed admission states are `admitted`, `deduplicated`, and
`capacity_dropped`; allowed gate states are `confirmed`, `rejected`,
`unverified`, and `not_gated`. Admitted rows require one of the three completed
gate states; deduplicated and capacity-dropped rows use `not_gated` and an
empty persisted event ID. An empty `rows` list is a valid observed
zero-candidate run. Gate directories alone are not a complete candidate audit:
they omit pre-gate queue drops and must not be used to claim zero duplicates.
`case_id` associates a candidate with one labeled clip. Unknown cases and
timestamps outside `[0,clip_duration_s]` are malformed; candidates in that clip
but outside every positive scenario-5 interval are valid false positives.
Overlapping scored scenario-5 incidents are ambiguous labels and invalidate
the input.

## Motion Acceptance Metrics

Report every numerator and denominator, not only percentages.

| Metric | Calculation and expected acceptance result |
| --- | --- |
| Person-detection recall | Detected visible person-frames / all visible labeled person-frames. Report by case; the pilot team must approve a numeric floor before treating it as a release gate. |
| ID switches | For each `person_ref`, count transitions between non-empty `observed_track_id` values across consecutive visible detections. Expected: zero in each controlled case, including across the temporary occlusion when the same ID is retained. |
| Scenario-5 recall | Positive `simultaneous_movement` intervals containing a candidate timestamp / all positive intervals. Expected: every positive interval hit. |
| Scenario-5 precision | Candidate timestamps matched to one positive interval / all scenario-5 candidates. In-window negatives and candidates outside every positive interval are false positives. Expected: every candidate matched; stationary crowd and jitter produce none. |
| Duplicate candidates | Additional matched candidates after the first in one labeled positive interval. Expected: zero. Report queue `deduped` separately because it proves suppression, not latch correctness. |
| Duplicate shown alerts | Additional persisted events after the first in one labeled incident. Expected: zero. |
| Detection delay | First matched candidate timestamp minus positive interval start. Report every value and median; no maximum has yet been approved. |
| Tracking visibility performance | Median `detect_batch.rate_per_s` from at least three paired hidden/shown runs, plus absolute FPS and percent delta. No numeric regression threshold has yet been approved. |

Scenario 4 passes only if its permitted moving person-frames are boxed, its
stationary/outside person-frames are not boxed, IDs remain stable, and it emits
zero candidates or alerts. Scenario 5 passes only if every positive interval
emits exactly one latched aggregate candidate, every negative interval emits
none, and every incident produces at most one shown alert. A TrueSight verdict
is reported separately from detector precision/recall.

## Motion Operator Workflow

### 1. Stage And Hash The Recordings

Start with [Portable Shell Setup](#portable-shell-setup), then:

```bash
export CHI_MOTION_DIR="$REPO_ROOT/data/chi_motion"
mkdir -p "$CHI_MOTION_DIR"
cp "/absolute/path/to/S4-P01.mp4" "$CHI_MOTION_DIR/S4-P01.mp4"
cp "/absolute/path/to/S4-N01.mp4" "$CHI_MOTION_DIR/S4-N01.mp4"
cp "/absolute/path/to/S5-P01.mp4" "$CHI_MOTION_DIR/S5-P01.mp4"
cp "/absolute/path/to/S5-N01.mp4" "$CHI_MOTION_DIR/S5-N01.mp4"
cp "/absolute/path/to/S5-P02.mp4" "$CHI_MOTION_DIR/S5-P02.mp4"
cp "/absolute/path/to/S5-P03.mp4" "$CHI_MOTION_DIR/S5-P03.mp4"
cp "/absolute/path/to/S4-P02.mp4" "$CHI_MOTION_DIR/S4-P02.mp4"
shasum -a 256 "$CHI_MOTION_DIR"/*.mp4
```

Create the manifest and two CSV files described above, review them, and make
them read-only for the run:

```bash
shasum -a 256 "$CHI_MOTION_DIR"/*.mp4 > "$CHI_MOTION_DIR/clips.sha256"
export CASE_CLIP="$CHI_MOTION_DIR/S5-P01.mp4"
"$PYTHON" - "$CASE_CLIP" <<'PY'
import cv2
import sys

capture = cv2.VideoCapture(sys.argv[1])
last_timestamp_s = None
while True:
    ok, _frame = capture.read()
    if not ok:
        break
    last_timestamp_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
capture.release()
if last_timestamp_s is None:
    raise SystemExit("clip has no decoded frames")
print(f"final decoded frame timestamp: {last_timestamp_s:.6f}")
PY
chmod 444 "$CHI_MOTION_DIR/clips.sha256" \
  "$CHI_MOTION_DIR/labels.csv" "$CHI_MOTION_DIR/observations.csv"
```

Run the timestamp block for every case and put its result in that case's
`clip_duration_s` cells. The absolute source paths are operator inputs.

### 2. Draw The Permitted Zone

For `S4-P01` and `S4-P02`, draw the real permitted polygon on the clip's
first frame and name it exactly `movement_permitted`:

```bash
export CASE_ID="S4-P02"
export CASE_CLIP="$CHI_MOTION_DIR/$CASE_ID.mp4"
export CASE_ZONES="$CHI_MOTION_DIR/$CASE_ID-zones.json"
"$PYTHON" tools/draw_zones.py --source "$CASE_CLIP" --out "$CASE_ZONES"
```

Left-click polygon vertices, right-click to finish, enter
`movement_permitted`, and press `s`. Do not fabricate a polygon for a clip
that does not visibly cross a meaningful operational boundary.

### 3. Build A Disposable One-Camera Site

Use an empty `CASE_ZONES` for whole-view cases. Scenario expectations live in
the separately named label columns, not in the runtime site file.

```bash
export CASE_ID="S5-P01"
export CASE_CLIP="$CHI_MOTION_DIR/$CASE_ID.mp4"
export CASE_ZONES=""
export RUN_ID="$(date +%Y%m%d-%H%M%S)"
export CHI_SITE="/private/tmp/chi-motion-$CASE_ID-$RUN_ID.json"
export CHI_OUT="$REPO_ROOT/runs/chi_motion/$CASE_ID-$RUN_ID"
test -f "$CASE_CLIP"
test -f "$CHI_MOTION_DIR/labels.csv"
mkdir -p "$CHI_OUT"

"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

camera = {
    "id": os.environ["CASE_ID"],
    "source": os.environ["CASE_CLIP"],
    "config": str(Path("configs/chi_pilot_v1.json").resolve()),
    "normal_movement": True,
    "multiple_people_moving": True,
    "movement_enter_speed_ratio": 0.05,
    "movement_exit_speed_ratio": 0.02,
    "movement_min_track_seconds": 0.4,
    "movement_min_people": 2,
    "movement_persistence_seconds": 0.5,
    "scene_context_mode": "manual",
    "environment_type": "public_space",
    "scene_description": "Operator-authored description of the recorded motion test.",
}
zones = os.environ.get("CASE_ZONES", "")
if zones:
    camera["zones"] = zones
    camera["permitted_movement_zones"] = ["movement_permitted"]
site = {
    "name": "Chi motion acceptance",
    "configured": True,
    "scene_context_policy": "auto",
    "cameras": [camera],
}
Path(os.environ["CHI_SITE"]).write_text(json.dumps(site, indent=2) + "\n")
PY
```

### 4. Run Real TrueSight And Exercise Both Overlay States

Confirm `gemma3:4b` is installed as described in
[Real TrueSight Acceptance Workflow](#real-truesight-acceptance-workflow).
In terminal 1, launch the operator app against the case:

```bash
MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON" -m cvti.app.shell \
  --site-config "$CHI_SITE" \
  --db "$CHI_OUT/events.db"
```

Start monitoring and leave the global **Show tracking** control off for the
hidden run. In terminal 2, run the engine:

```bash
MPLCONFIGDIR="$MPLCONFIGDIR" OLLAMA_API_KEY=ollama "$PYTHON" -m cvti.serving.pipeline \
  --site-config "$CHI_SITE" \
  --gate-provider ollama \
  --gate-model gemma3:4b \
  --gate-base-url http://127.0.0.1:11434/v1 \
  --mapper-provider ollama \
  --mapper-model gemma3:4b \
  --mapper-base-url http://127.0.0.1:11434/v1 \
  --gate-sensitivity balanced \
  --notify console \
  --output-dir "$CHI_OUT" \
  --target-fps 5 \
  --publish-fps 24 \
  --imgsz 640 \
  --seconds 30 \
  --gate-drain 180 \
  --mobile-port 8710 2>&1 | tee "$CHI_OUT/operator.log"
```

During the shown run, turn on global **Show tracking**, verify the tile
reconnects with `tracking=true`, then test that one camera's **Hide** and
**Use global** override only that tile. Confirm the detection engine remains
running and candidate counts do not change merely because visibility changed.

For performance, use three independent hidden runs and three independent shown
runs in alternating order after model warm-up. Keep all other inputs fixed,
use a fresh output directory for each run, and run no tests or unrelated
workloads concurrently. Start the pipeline command above in the background,
then attach exactly one capture client while it runs:

```bash
export PAIR_NUMBER="1"
export PAIR_ID="pair-$PAIR_NUMBER"
export TRACKING_MODE="hidden"  # alternate hidden, shown for pair 1, 2, and 3
export TRACKING_QUERY="0"      # hidden=0; shown=1
export PERF_RUN_ID="$PAIR_ID-$TRACKING_MODE"
export PAIR_RUN="$TRACKING_MODE-$PAIR_NUMBER"
export CHI_OUT="$REPO_ROOT/runs/chi_motion/perf-$PAIR_RUN"
export CAPTURE_PATH="$CHI_MOTION_DIR/captures/$PAIR_RUN.mjpeg"
export PERF_PATH="$CHI_MOTION_DIR/perf/$PAIR_RUN.json"
mkdir -p "$CHI_OUT" "$CHI_MOTION_DIR/perf" "$CHI_MOTION_DIR/captures"

MPLCONFIGDIR="$MPLCONFIGDIR" OLLAMA_API_KEY=ollama "$PYTHON" -m cvti.serving.pipeline \
  --site-config "$CHI_SITE" \
  --gate-provider ollama \
  --gate-model gemma3:4b \
  --gate-base-url http://127.0.0.1:11434/v1 \
  --mapper-provider ollama \
  --mapper-model gemma3:4b \
  --mapper-base-url http://127.0.0.1:11434/v1 \
  --gate-sensitivity balanced \
  --notify console \
  --output-dir "$CHI_OUT" \
  --target-fps 5 \
  --publish-fps 24 \
  --imgsz 640 \
  --seconds 30 \
  --gate-drain 180 \
  --mobile-port 8710 >"$CHI_OUT/operator.log" 2>&1 &
ENGINE_PID=$!
while test ! -s "$CHI_OUT/frames.json"; do sleep 0.2; done
read FRAME_PORT FRAME_TOKEN <<EOF
$("$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["port"], d["token"])' "$CHI_OUT/frames.json")
EOF
curl --silent --show-error --max-time 12 \
  "http://127.0.0.1:$FRAME_PORT/stream/$CASE_ID?tracking=$TRACKING_QUERY&token=$FRAME_TOKEN" \
  --output "$CAPTURE_PATH"
CAPTURE_STATUS="$?"
test "$CAPTURE_STATUS" -eq 0 -o "$CAPTURE_STATUS" -eq 28
wait "$ENGINE_PID"
cp "$CHI_OUT/perf_report.json" "$PERF_PATH"

export CLIP_SHA256="$(shasum -a 256 "$CASE_CLIP" | awk '{print $1}')"
export CONFIG_SHA256="$(shasum -a 256 "$CHI_SITE" | awk '{print $1}')"
export CAPTURE_SHA256="$(shasum -a 256 "$CAPTURE_PATH" | awk '{print $1}')"
"$PYTHON" - "$PERF_PATH" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
engine = report["stages"]["detect_batch"]["engine"]
report["chi_motion_performance"] = {
    "schema_version": 1,
    "case_id": os.environ["CASE_ID"],
    "clip_sha256": os.environ["CLIP_SHA256"],
    "pair_id": os.environ["PAIR_ID"],
    "run_id": os.environ["PERF_RUN_ID"],
    "tracking_mode": os.environ["TRACKING_MODE"],
    "tracking_query": int(os.environ["TRACKING_QUERY"]),
    "config_sha256": os.environ["CONFIG_SHA256"],
    "sample_duration_s": engine["span_s"],
    "sample_count": engine["count"],
    "capture_path": os.environ["CAPTURE_PATH"],
    "capture_sha256": os.environ["CAPTURE_SHA256"],
}
path.write_text(json.dumps(report, indent=2) + "\n")
PY
```

The client timeout is expected if the finite clip has not ended. Keep the six
MJPEG captures as proof that each report had an active raw (`tracking=0`) or
shown (`tracking=1`) viewer. Repeat the complete block for hidden and shown in
each of `pair-1`, `pair-2`, and `pair-3`, changing both `TRACKING_MODE` and
`TRACKING_QUERY` together. The scorer hashes the capture named by each report
and requires six distinct, non-empty captures. It also requires unique run IDs,
exactly matched pair IDs, one case and clip hash, one config hash, correct mode
and query, schema version 1, sample count/duration equal to the detection stage,
and equal sample count plus duration within 1 ms inside each pair. A single run,
reused report or capture, run without an active viewer, or run under competing
load is not a reproducible measurement. Exercise global **Show tracking** and
the per-camera **Hide** / **Use global** controls separately in the app; those
controls select the same stream query and do not change detection.

### 5. Export The Candidate Audit

Scenario-5 acceptance requires an audit hook that records every generated
candidate before `AlertQueue.add()`, updates admission to `admitted`,
`deduplicated`, or `capacity_dropped`, and appends the eventual gate status and
persisted event ID. The current build has that retained table for concealment,
but not for scenario 5. Until the same hook is enabled for
`multiple_people_moving`, this acceptance input and all candidate-derived
metrics remain **unmeasured**. Do not reconstruct it from gate directories.

When the capture hook writes the schema above to the run's
`motion_candidate_audit` table, either pass `events.db` directly to the scorer
or export its rows with structured SQLite and JSON parsing:

```bash
export CHI_AUDIT_DB="$CHI_OUT/events.db"
export CHI_AUDIT_JSON="$CHI_MOTION_DIR/motion-candidate-audit.json"
"$PYTHON" - "$CHI_AUDIT_DB" "$CHI_AUDIT_JSON" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
con = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
try:
    rows = [dict(row) for row in con.execute(
        "SELECT candidate_id, case_id, timestamp_s, admission_status, "
        "gate_status, persisted_event_id "
        "FROM motion_candidate_audit ORDER BY rowid"
    )]
finally:
    con.close()
target.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
print(f"exported {len(rows)} rows to {target}")
PY
```

For an external capture hook that already emits JSON, combine all case rows in
one `rows` list and retain the original files and hashes. An empty list is valid
only when the hook was active for the complete run.

### 6. Score And Retain The Result

Complete `observations.csv` by reviewing the raw and tracking captures at the
configured 5 FPS, then run:

```bash
"$PYTHON" tools/score_chi_motion.py \
  --labels "$CHI_MOTION_DIR/labels.csv" \
  --observations "$CHI_MOTION_DIR/observations.csv" \
  --audit "$CHI_AUDIT_JSON" \
  --hidden-perf "$CHI_MOTION_DIR/perf/hidden-1.json" \
                "$CHI_MOTION_DIR/perf/hidden-2.json" \
                "$CHI_MOTION_DIR/perf/hidden-3.json" \
  --shown-perf "$CHI_MOTION_DIR/perf/shown-1.json" \
               "$CHI_MOTION_DIR/perf/shown-2.json" \
               "$CHI_MOTION_DIR/perf/shown-3.json" \
  --output "$CHI_MOTION_DIR/chi-motion-score.json"
"$PYTHON" -m json.tool "$CHI_MOTION_DIR/chi-motion-score.json" >/dev/null
shasum -a 256 "$CHI_MOTION_DIR/chi-motion-score.json"
```

The tool fails with exit 2 for missing, incomplete, ambiguous, or malformed
inputs. Its retained schema-version-2 JSON hashes labels, observations, audit,
all six reports, and all six capture files. It records expected/observed sample
coverage, person recall counts, ID switches, scenario-4 observation checks,
per-candidate positive/negative/out-of-window classification, scenario-5
precision/recall, admission/gate counts, duplicate candidates and persisted
alerts per incident, detection delay, all three matched performance pairs, and
the median hidden/shown FPS impact. Pass the SQLite database as
`--audit "$CHI_AUDIT_DB"` to skip the JSON export.

## Local Motion Replay - 2026-09-13

The most representative retained clip was replayed through the production
pipeline with real local TrueSight and an authenticated tracking viewer. The
sanitized facts, source and transient-artifact hashes, runtime versions, and
exact Ollama model digest are retained in
[`docs/evidence/chi-motion-local-replay-2026-09-13.json`](evidence/chi-motion-local-replay-2026-09-13.json):

- Clip: `data/test_clips/normal_street_01.mp4`
- SHA-256: `2ca3549f53d02f645106edadf97f52f3c0760d1f9b07e32e5befba6df1c894c8`
- Media: 640x360, 25 FPS, 13.64 seconds
- Config: both motion scenarios enabled, default thresholds, two-person
  threshold, whole view permitted
- Tracking stream: 205,828 MJPEG bytes received over an authenticated
  `tracking=1` connection before its intentional 8-second client timeout
- Candidate: timestamp 10.0s, tracks 35 and 52, spatially separated group box,
  one candidate admitted and one additional queue duplicate suppressed
- TrueSight: one verified and confirmed result, confidence 0.90, zero errors,
  zero unverified results; one non-provisional event persisted
- Replay output reported three gate frames, 17 event frames, a subject crop,
  replay clip, alert JSON, and verification JSON; their hashes or aggregate
  manifest hash are retained, while the transient originals are not
- Performance observation: 48 detection samples in a 14.595s retained window
  (`detect_batch.rate_per_s=3.289`), 108 decoded samples in 20.037s
  (`decode.rate_per_s=5.39`); first inference was 5.329s

This is a plumbing smoke result, not a Chi acceptance result.
`normal_street_01.mp4` had no frozen pre-run frame/identity labels, no
controlled negative companion, and no permitted-zone boundary. Therefore
person-detection recall, ID switches, scenario-5 precision/recall, formal
detection delay, stationary-crowd behavior, temporary-occlusion behavior, and
zone-boundary behavior remain **unmeasured**. Persisted duplicate alerts were
observed as 0, but the queue's one suppressed duplicate must be investigated
with a dedicated candidate audit before claiming zero duplicate candidates.

Hidden-versus-shown FPS is also **unmeasured**. The shown run overlapped another
validation workload and there is no overlay-specific timing series; presenting
its single throughput sample against a separate run would not be reproducible.
Use the paired protocol above.

## Repaired End-To-End Path

```text
camera frame
  -> shared YOLO detections (people and COCO personal bags)
  -> per-camera pose inference on heavy_stride samples
  -> per-person bag assignment and temporal ConcealmentDetector
  -> concealment RawEvent with destination, score, components, reasons,
     limited-evidence flag, track ID, subject box, and associated bag box
  -> CandidateAlert metadata after CustomizationEngine plus reviewed Agent
     Mapper scene compatibility
  -> deduplicated shared alert queue with video and wall-clock timestamps
  -> TrueSight with 3 chronological full frames plus the subject crop
  -> retained SQLite concealment_audit row for every generated candidate,
     including queue admission outcome and any eventual gate verdict
  -> confirmed-only event persistence, notification, and UI evidence
```

Skipped pose frames do not erase temporal history. `expire(timestamp)` still
runs every video frame and removes stale tracks after the grace period. Personal
bags are limited to COCO backpack, handbag, and suitcase classes and are
assigned to one nearby pose track. The physical-bag owner remains stable across
near-equal frame jitter for at least the concealment scoring window; an initial
exact ownership tie is left unassigned.
Shopping baskets and trolleys are not concealment destinations.

The pose heuristic is a recall-oriented candidate generator. It uses the
existing `0.63` score threshold and four consecutive above-threshold sampled
observations. It does not identify a product or decide theft. A reviewed retail
scene makes the rule applicable, and TrueSight makes the final visible-evidence
decision.

## Scenario 10 Recording Matrix

Record seven separate 10-second clips from the same fixed Chi pilot camera.
Use one actor, one ordinary retail product, stable lighting, and an already
reviewed Agent Mapper context of `retail_shop` with a merchandise or checkout
zone active. Keep the actor's shoulders, wrists, hips, destination, and product
visible. Times are relative to the start of each clip.

If the performed action misses a planned boundary, label the observed interval
before running the system and retain that actual interval with the result. Do
not move a label after seeing a verdict.

| ID | Case and labeled intervals | Candidate expectation | TrueSight expectation | Accepted end state |
| --- | --- | --- | --- | --- |
| `S10-P01` | **Pocket positive:** `0.0-2.0s` neutral baseline; `2.0-4.0s` reach to merchandise; `4.0-7.0s` retract to pocket/waistband and leave product concealed; `7.0-10.0s` hands away. | At least one `concealment` candidate in `4.0-8.0s`, `destination=waist`, with destination, retract, and dwell evidence. | `confirmed=true`; reason names visible pocket/waist concealment. | Exactly one non-provisional, verified shoplifting event is persisted and shown with full-frame evidence, subject crop, and replay clip. |
| `S10-P02` | **Bag positive:** personal bag remains visible; `0.0-2.0s` neutral baseline; `2.0-4.0s` reach to merchandise; `4.0-7.0s` retract and place product inside the personal bag; `7.0-10.0s` hands away. | At least one candidate in `4.0-8.0s`, `destination=bag`, with a non-null associated bag assigned only to the actor. | `confirmed=true`; reason names visible placement into a personal bag. | Exactly one non-provisional, verified shoplifting event is persisted and shown with evidence and replay clip. |
| `S10-N01` | **Trolley safe:** trolley/basket visible; `0.0-2.0s` baseline; `2.0-4.0s` reach; `4.0-7.0s` place product in trolley/basket; `7.0-10.0s` continue shopping. | No concealment candidate and no personal-bag destination. | If an upstream candidate nevertheless reaches the gate, `confirmed=false`. | No confirmed or unverified shoplifting event is persisted or shown. |
| `S10-N02` | **Phone to pocket:** `0.0-2.0s` show phone openly; `2.0-4.0s` use it at chest height; `4.0-5.5s` put phone in pocket; `5.5-10.0s` hands away. | No persistent concealment candidate. | If gated, `confirmed=false`; reason identifies phone handling rather than merchandise concealment. | No confirmed or unverified shoplifting event. |
| `S10-N03` | **Clothing adjustment:** no product held; `0.0-2.0s` baseline; `2.0-4.0s` briefly adjust pocket/waist clothing; `4.0-10.0s` neutral. | No persistent concealment candidate. | If gated, `confirmed=false`; reason identifies clothing adjustment or lack of merchandise. | No confirmed or unverified shoplifting event. |
| `S10-N04` | **Browsing:** `0.0-2.0s` baseline; `2.0-7.0s` examine merchandise at shelf height without moving it to body/bag; `7.0-10.0s` replace it or keep browsing. | No concealment candidate. | If gated, `confirmed=false`; reason identifies ordinary browsing. | No confirmed or unverified shoplifting event. |
| `S10-N05` | **Open carry:** `0.0-2.0s` baseline; `2.0-4.0s` take product; `4.0-10.0s` keep it continuously visible while walking or standing. | No concealment candidate. | If gated, `confirmed=false`; reason identifies openly carried goods. | No confirmed or unverified shoplifting event. |

`UNVERIFIED` is not a passing positive or negative verdict. It means the gate
did not complete and must be reported as an infrastructure error. A normal
negative that never generates a candidate has no gate verdict; record it as
`not_gated`, not as a TrueSight rejection.

## Result Record

Retain one row per case with these fields:

| Field | Required value |
| --- | --- |
| Identity | case ID, clip path, SHA-256, camera ID, recording date |
| Labels | actual reach interval, destination-action interval, expected class |
| Context | mapper lifecycle, reviewed status, environment, active zone roles |
| Candidate | all generation-ordered rows with queue outcome; timestamp-sorted rows; inclusive-window rows; for positives, first in-window timestamp or `none`; destination, in-window peak score, components, limited flag, associated bag, track ID |
| Gate | `confirmed`, `rejected`, `not_gated`, or `unverified`; confidence, reason, prompt fingerprint, model tag and digest |
| Delivery | event ID, duplicate count, evidence directory, full frames present, subject crop present, replay clip present, UI visibility |
| Performance | detection delay from destination-action start and pose-stage FPS |

Compute candidate recall over `S10-P01` and `S10-P02`; post-gate precision and
recall over all gated cases; duplicate alerts per incident; positive detection
delay; and pose-stage throughput. Report numerators and denominators alongside
each rate. With only two positive cases, percentages are smoke evidence, not a
general accuracy estimate.

## Acceptance Gate

Scenario 10 passes only when all of the following are true in one recorded run:

1. `S10-P01` and `S10-P02` both generate the expected candidate and receive a
   completed `confirmed=true` TrueSight verdict.
2. Each positive produces exactly one persisted event whose UI evidence includes
   the scene, the subject crop, and a playable replay clip.
3. No negative case produces a confirmed or fail-visible unverified shoplifting
   event. Any negative candidate that is gated must receive `confirmed=false`.
4. Bag evidence belongs only to the correct actor; trolley/basket placement has
   no personal-bag destination.
5. The prompt measurement is either completed on the restored frozen corpus or
   remains explicitly labeled `unmeasured`; previous-prompt metrics are never
   presented as current results.

## Portable Shell Setup

Run commands from a repository checkout. A normal checkout uses its own
`.venv`:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp}"
cd "$REPO_ROOT"
test -x "$PYTHON"
```

For this linked worktree only, the source checkout owns the shared environment.
Set this before `test -x "$PYTHON"` when the worktree has no `.venv`:

```bash
export PYTHON="/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python"
```

For a new checkout without an environment, create it first. The desktop UI
dependencies are listed separately because they are not in the
[Python requirements](../requirements.txt):

```bash
python3 -m venv "$REPO_ROOT/.venv"
export PYTHON="$REPO_ROOT/.venv/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$REPO_ROOT/requirements.txt"
"$PYTHON" -m pip install PyQt6 PyQt6-WebEngine
```

## Automated Mock And Stub Smoke

This command uses synthetic poses and mocked/stubbed model and provider calls.
It proves the concealment, serving, prompt, and evidence contracts without
running a real Chi clip or asking TrueSight to judge one:

```bash
MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON" -m pytest tests/test_concealment.py tests/test_heavy_models_earn_their_frames.py tests/test_serving.py tests/test_prompt_regression.py tests/test_gate_evidence_quality.py tests/test_alert_sink.py -q
```

The prompt fingerprint/status check is also non-measuring:

```bash
MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON" tools/prompt_regression.py check
```

Do not use `--gate-provider mock` for acceptance. The engine normally refuses
that provider, and the development override confirms candidates without looking
at the images. A mock run demonstrates plumbing only.

## Real TrueSight Acceptance Workflow

The production CLI accepts file paths through each camera's `source` field; it
does not expose a clip-upload endpoint. The steps below therefore submit one
clip at a time by creating a disposable one-camera site config. Running one case
per output directory keeps its context, gate counts, database, and evidence
unambiguous.

### 1. Name And Stage The Clips

Create the local, uncommitted clip directory and place the seven recordings at
these exact paths:

```bash
export CHI_CLIP_DIR="$REPO_ROOT/data/chi_s10"
mkdir -p "$CHI_CLIP_DIR"
cp "/absolute/path/to/pocket-positive.mp4" "$CHI_CLIP_DIR/S10-P01-pocket-positive.mp4"
cp "/absolute/path/to/bag-positive.mp4" "$CHI_CLIP_DIR/S10-P02-bag-positive.mp4"
cp "/absolute/path/to/trolley-safe.mp4" "$CHI_CLIP_DIR/S10-N01-trolley-safe.mp4"
cp "/absolute/path/to/phone-to-pocket.mp4" "$CHI_CLIP_DIR/S10-N02-phone-to-pocket.mp4"
cp "/absolute/path/to/clothing-adjustment.mp4" "$CHI_CLIP_DIR/S10-N03-clothing-adjustment.mp4"
cp "/absolute/path/to/browsing.mp4" "$CHI_CLIP_DIR/S10-N04-browsing.mp4"
cp "/absolute/path/to/open-carry.mp4" "$CHI_CLIP_DIR/S10-N05-open-carry.mp4"
shasum -a 256 "$CHI_CLIP_DIR"/*.mp4
```

The `/absolute/path/to/...` values are operator inputs, not repository paths.
Retain the resulting hashes with the matrix labels.

### 2. Start The Local Provider

Install the model once:

```bash
ollama pull gemma3:4b
```

In terminal 1, start Ollama and leave it running. If it is already running, do
not start a second server.

```bash
ollama serve
```

From another terminal, verify the provider and model before each acceptance
session:

```bash
export MODEL_TAGS="${CHI_MOTION_DIR:-$CHI_CLIP_DIR}/ollama-tags.json"
curl -fsS http://127.0.0.1:11434/api/tags --output "$MODEL_TAGS"
"$PYTHON" - "$MODEL_TAGS" <<'PY'
import json
import sys

models = json.load(open(sys.argv[1]))["models"]
match = next(model for model in models if model["name"] == "gemma3:4b")
print(match["digest"])
PY
ollama list
```

The output must include `gemma3:4b`. Retain `ollama-tags.json` and its printed
digest with each acceptance session.

### 3. Select One Case And Build Its Site Config

Repeat steps 3-8 for every row in the matrix. Set the case and clip for the
current run; this example selects the pocket-positive case:

```bash
export CASE_ID="S10-P01"
export CASE_CLIP="$CHI_CLIP_DIR/S10-P01-pocket-positive.mp4"
export EXPECTED_CLASS="positive"
export ACTION_WINDOW_START_S="4.0"
export ACTION_WINDOW_END_S="8.0"
export RUN_ID="$(date +%Y%m%d-%H%M%S)"
export CHI_SITE="/private/tmp/chi-${CASE_ID}-${RUN_ID}.json"
export CHI_OUT="$REPO_ROOT/runs/chi_s10/${CASE_ID}-${RUN_ID}"
test -f "$CASE_CLIP"
mkdir -p "$CHI_OUT"
```

Use the filename table below for the other six runs:

Set all five inputs for the selected row before generating its config. The
window is inclusive and must match the observed action label fixed before the
run; update these values if the retained observed interval differs from the
planned interval in the recording matrix.

| Case | `CASE_CLIP` basename | `EXPECTED_CLASS` | `ACTION_WINDOW_START_S` | `ACTION_WINDOW_END_S` |
| --- | --- | --- | --- | --- |
| `S10-P01` | `S10-P01-pocket-positive.mp4` | `positive` | `4.0` | `8.0` |
| `S10-P02` | `S10-P02-bag-positive.mp4` | `positive` | `4.0` | `8.0` |
| `S10-N01` | `S10-N01-trolley-safe.mp4` | `negative` | `4.0` | `7.0` |
| `S10-N02` | `S10-N02-phone-to-pocket.mp4` | `negative` | `4.0` | `5.5` |
| `S10-N03` | `S10-N03-clothing-adjustment.mp4` | `negative` | `2.0` | `4.0` |
| `S10-N04` | `S10-N04-browsing.mp4` | `negative` | `2.0` | `7.0` |
| `S10-N05` | `S10-N05-open-carry.mp4` | `negative` | `4.0` | `10.0` |

Generate a config using the production site schema. `require_reviewed` makes
Agent Mapper output wait for operator approval. The accepted `merchandise` role
makes the [retail shoplifting rule](../configs/retail_pipeline_v1.json)
applicable after the scene is reviewed. See
[Agent Mapper Operations](AGENT_MAPPER_OPERATIONS.md) for lifecycle and recovery
details.

```bash
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["REPO_ROOT"])
case_id = os.environ["CASE_ID"]
site = {
    "name": f"Chi scenario 10 - {case_id}",
    "configured": True,
    "notify": "console",
    "scene_context_policy": "require_reviewed",
    "cameras": [{
        "id": case_id,
        "source": os.environ["CASE_CLIP"],
        "config": str(root / "configs" / "retail_pipeline_v1.json"),
        "concealment": True,
        "heavy_stride": 2,
        "accepted_zone_roles": ["merchandise"],
    }],
}
Path(os.environ["CHI_SITE"]).write_text(json.dumps(site, indent=2) + "\n")
PY
"$PYTHON" -c 'import json, os; from cvti.serving.camera import load_site_config; site=load_site_config(os.environ["CHI_SITE"]); assert len(site["cameras"]) == 1; print(site["cameras"][0])'
```

### 4. Map The Camera With The Production Preflight

Run the serving layer's synchronous mapping contract before scoring the short
file. This writes the same site-scoped context, representative frame, source
fingerprint, and lifecycle artifacts that engine startup consumes:

```bash
MPLCONFIGDIR="$MPLCONFIGDIR" OLLAMA_API_KEY=ollama "$PYTHON" - <<'PY'
import json
import os

from cvti.serving.camera import load_site_config
from cvti.serving.pipeline import prepare_scene_mapping

site = load_site_config(os.environ["CHI_SITE"])
result = prepare_scene_mapping(
    site,
    output_dir=os.environ["CHI_OUT"],
    gate_provider="ollama",
    gate_model="gemma3:4b",
    gate_base_url="http://127.0.0.1:11434/v1",
    mapper_provider="ollama",
    mapper_model="gemma3:4b",
    mapper_base_url="http://127.0.0.1:11434/v1",
)
print(json.dumps(result.statuses, indent=2))
PY
```

The case must report `ready_unreviewed`, and these paths must exist:

```bash
test -f "$CHI_OUT/context/$CASE_ID/scene_context.json"
test -f "$CHI_OUT/context/$CASE_ID/source_frame.jpg"
test -f "$CHI_OUT/context/$CASE_ID/mapping_status.json"
cat "$CHI_OUT/context/$CASE_ID/mapping_status.json"
```

A mapping failure or missing representative frame blocks the case; record it as
an infrastructure error instead of changing the label or enabling mock.

### 5. Approve And Associate The Scene

In terminal 2, launch the operator application against the generated site and
its case-specific database:

```bash
cd "$REPO_ROOT"
MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON" -m cvti.app.shell \
  --site-config "$CHI_SITE" \
  --db "$CHI_OUT/events.db"
```

Create/sign in as an owner or installer if prompted. In **Scene review** (or the
camera's **Rules** scene panel), select `$CASE_ID`, inspect the mapper's saved
representative frame, correct the environment to `retail_shop` if necessary,
retain a truthful scene description, and approve it. Do not accept a suggested
zone for this test: the config already associates the camera with the accepted
`merchandise` role, while mapper suggestions deliberately remain inert.

Confirm the context is approved for this camera and source:

```bash
cat "$CHI_OUT/context/$CASE_ID/mapping_status.json"
```

The lifecycle must now be `ready_reviewed`. Leave the app running for the live
wall and evidence inspection.

### 6. Run The Clip Through Real TrueSight

In terminal 3, run the production serving path. The file is processed once and
then the gate drains before shutdown; this avoids repeated incidents changing
the case denominator.

```bash
cd "$REPO_ROOT"
set -o pipefail
MPLCONFIGDIR="$MPLCONFIGDIR" OLLAMA_API_KEY=ollama "$PYTHON" -m cvti.serving.pipeline \
  --site-config "$CHI_SITE" \
  --gate-provider ollama \
  --gate-model gemma3:4b \
  --gate-base-url http://127.0.0.1:11434/v1 \
  --mapper-provider ollama \
  --mapper-model gemma3:4b \
  --mapper-base-url http://127.0.0.1:11434/v1 \
  --gate-sensitivity balanced \
  --notify console \
  --output-dir "$CHI_OUT" \
  --target-fps 5 \
  --imgsz 640 \
  --seconds 30 \
  --gate-drain 180 \
  --mobile-port 8710 2>&1 | tee "$CHI_OUT/operator.log"
```

This exact run uses the reviewed mapper cache, the configured retail rule, the
shared pose/concealment path, local TrueSight, the alert sink, and the live UI.

### 7. Extract Candidate, TrueSight, And Performance Outcomes

Watch terminal 3 and retain `operator.log`. The structured production record is
the `concealment_audit` table in `$CHI_OUT/events.db`. The alert sink inserts one
row for every generated concealment candidate before queue admission, records
`admitted`, `deduplicated`, or `capacity_dropped`, and later attaches a
`confirmed`, `rejected`, or `unverified` verdict to admitted rows that reach
TrueSight. An empty table means no concealment candidate was generated only
when `operator.log` also contains no concealment audit error. It is never
evidence that TrueSight rejected the clip. These rows follow the site's normal
retention period and emergency disk purge.

The engine exits after the file ends and its queued verdicts drain. Its final
lines report `alerts_queued` and gate `verified`, `confirmed`, `rejected`,
`errors`, `unverified`, and `deduped` counts. Extract the case-specific audit
health and queue lines with:

```bash
grep -E 'ready_reviewed|POSSIBLE CONCEALMENT|CONFIRMED|REJECTED|UNVERIFIED|alerts_queued|gate=' "$CHI_OUT/operator.log"
```

Only the two positive cases may end in `[CONFIRMED]`. Any `UNVERIFIED`, gate
error, or breaker-open result invalidates the case.

After shutdown, run this exact extractor. It retains all audit rows for the
current camera in candidate-generation order, sorts a separate view by clip-relative
candidate timestamp, filters an inclusive copy to the explicit labeled action
window, and derives positive recall/delay only from that filtered copy. It also
reads sampled pose-stage throughput from the final
`perf_report.json`. It writes the reusable case artifact
`$CHI_OUT/scenario10_result.json`.

```bash
"$PYTHON" - <<'PY'
import json
import os
import sqlite3
from pathlib import Path

case_id = os.environ["CASE_ID"]
expected_class = os.environ["EXPECTED_CLASS"]
if expected_class not in {"positive", "negative"}:
    raise SystemExit("EXPECTED_CLASS must be 'positive' or 'negative'")
window_start = float(os.environ["ACTION_WINDOW_START_S"])
window_end = float(os.environ["ACTION_WINDOW_END_S"])
if window_start > window_end:
    raise SystemExit("ACTION_WINDOW_START_S must be <= ACTION_WINDOW_END_S")
out = Path(os.environ["CHI_OUT"])
con = sqlite3.connect(out / "events.db")
con.row_factory = sqlite3.Row
try:
    all_rows = [dict(row) for row in con.execute(
        "SELECT * FROM concealment_audit WHERE camera_id = ? ORDER BY id",
        (case_id,),
    )]
finally:
    con.close()
for row in all_rows:
    row["components"] = json.loads(row.pop("components_json"))
    bag_json = row.pop("associated_bag_json")
    row["associated_bag"] = json.loads(bag_json) if bag_json is not None else None
    row["reasons"] = json.loads(row.pop("reasons_json"))
required = {
    "candidate_timestamp", "track_id", "destination", "peak_score",
    "components", "limited", "associated_bag", "enqueued_at", "verdict_at",
    "gate_result_timestamp", "gate_latency_s", "verdict", "confirmed",
    "confidence", "reason", "gate_error", "prompt_version", "admission_status",
}
for row in all_rows:
    missing = sorted(required - row.keys())
    if missing:
        raise SystemExit(f"incomplete concealment audit row: {missing}")

chronological = sorted(all_rows, key=lambda row: row["candidate_timestamp"])
window_rows = [
    row for row in chronological
    if window_start <= row["candidate_timestamp"] <= window_end
]
first_in_window = window_rows[0] if window_rows else None
detection_delay = (
    round(first_in_window["candidate_timestamp"] - window_start, 3)
    if expected_class == "positive" and first_in_window else None
)
scored_rows = window_rows if expected_class == "positive" else chronological

perf = json.loads((out / "perf_report.json").read_text())
pose = perf.get("stages", {}).get("pose_infer", {}).get(case_id)
record = {
    "case_id": case_id,
    "expected_class": expected_class,
    "labeled_action_window": {
        "start_s": window_start, "end_s": window_end, "inclusive": True,
    },
    "candidate_count": len(all_rows),
    "admitted_candidate_count": sum(
        row["admission_status"] == "admitted" for row in all_rows
    ),
    "deduplicated_candidate_count": sum(
        row["admission_status"] == "deduplicated" for row in all_rows
    ),
    "capacity_dropped_candidate_count": sum(
        row["admission_status"] == "capacity_dropped" for row in all_rows
    ),
    "window_candidate_count": len(window_rows),
    "candidate_recall_hit": bool(window_rows) if expected_class == "positive" else None,
    "first_in_window_candidate_timestamp": (
        first_in_window["candidate_timestamp"] if first_in_window else None
    ),
    "detection_delay_s": detection_delay,
    "peak_score": max((row["peak_score"] for row in scored_rows), default=None),
    "all_candidates": all_rows,
    "chronological_candidates": chronological,
    "window_candidates": window_rows,
    "pose_stage": pose,
    "pose_stage_fps": pose.get("rate_per_s") if pose else None,
}
target = out / "scenario10_result.json"
target.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
```

Each generated candidate row contains `candidate_timestamp`, `track_id`, `destination`,
`peak_score`, `components`, `limited`, `associated_bag`, `reasons`,
`admission_status`, `enqueued_at`, `verdict_at`, `gate_result_timestamp`,
`gate_latency_s`, verdict, confidence/reason/error, and `prompt_version`.
Verdict fields are null for deduplicated or capacity-dropped rows.
`candidate_timestamp` is seconds
from the clip's decoder timeline. `gate_latency_s` is queue entry through sink
receipt. `pose_stage_fps` is the observed pose invocation rate over the retained
performance window; retain `count`, `per_unit_ms`, `p50_ms`, and `p95_ms` with
it. A null rate from a one-sample window is insufficient performance evidence.

For positives, use `first_in_window_candidate_timestamp` and
`detection_delay_s`; `candidate_recall_hit` is true only when the inclusive
labeled window contains a candidate. Use the maximum in-window `peak_score` and
its row for components, destination, limited status, associated bag, and track
ownership. `all_candidates` preserves candidate-generation order for
false-positive and duplicate analysis, while `chronological_candidates` and
`window_candidates` are sorted by candidate timestamp.

For negatives, `candidate_recall_hit` and `detection_delay_s` are always null;
their explicit action windows support analysis but can never satisfy positive
recall. Zero `all_candidates` means no candidate was generated. If generated
rows were admitted, every completed verdict must say `rejected`; deduplicated
and capacity-dropped rows have no verdict. Any `unverified` row invalidates the
case. Candidate recall is the
number of positive case IDs with `candidate_recall_hit=true` divided by two,
not the number of confirmed SQLite events.

### 8. Retrieve Persistence, Replay, And UI Evidence

Query the actual SQLite event store without requiring the external `sqlite3`
program:

```bash
"$PYTHON" - <<'PY'
import json
import os
import sqlite3

db = os.path.join(os.environ["CHI_OUT"], "events.db")
con = sqlite3.connect(db)
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT id, camera_id, rule, confidence, reason, evidence_dir, "
    "unverified, prompt_version FROM events "
    "WHERE camera_id = ? AND rule = 'shoplifting' ORDER BY id",
    (os.environ["CASE_ID"],),
).fetchall()
print(json.dumps([dict(row) for row in rows], indent=2))
con.close()
PY
```

For `S10-P01` and `S10-P02`, exactly one row must be present with
`unverified=0`, a non-empty `prompt_version`, and an evidence directory. Verify
the replay and stills directly:

```bash
find "$CHI_OUT/events" -type f \( -name 'event.json' -o -name 'subject.jpg' -o -name 'clip.mp4' -o -name 'frame_*.jpg' \) -print
```

The positive evidence directory must contain `event.json`, `subject.jpg`,
`clip.mp4`, and chronological `frame_*.jpg` files. In the still-running desktop
app, open **Alerts**, select the case's event, and verify that the subject image
points to the actor and the event replay plays through the labeled action. This
UI action exercises
[`ConsoleBackend.event_clip()`](../cvti/app/console_backend.py), the
application's supported replay contract.

For negative cases, an empty query result is correct when no candidate was
gated or when a high-priority shoplifting candidate was rejected. Use
`scenario10_result.json`, not the absence of a SQLite row, to distinguish
`rejected` from `not_gated`; neither outcome may leave a confirmed or
unverified event in the UI. Rejected and unverified high-priority candidates
are audit records only and do not create user-facing alerts.

After recording the matrix row, close the app and repeat from step 3 with the
next case ID and filename. Keep every timestamped output directory.

## Full Python Regression

No full-suite rerun is needed for a recorded clip, but the repository-wide
command is:

```bash
MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON" -m pytest
```

The automated commands validate implementation contracts. They do not replace
the seven real TrueSight runs or measure the changed prompt.

## Prompt Measurement When The Corpus Returns

First verify that the restored golden directory contains its manifest, cases,
and frames. Start Ollama with the pilot model, then run a complete fresh replay:

```bash
MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON" tools/prompt_regression.py run --golden-dir runs/eval/golden --gate-provider ollama --gate-model gemma3:4b --sensitivity balanced --fresh --verbose
```

Only a complete replay with zero errored cases is a measurement. Review the
metrics and model digest before deliberately rerunning with
`--update-baseline`; partial or errored runs must remain non-baseline results.
