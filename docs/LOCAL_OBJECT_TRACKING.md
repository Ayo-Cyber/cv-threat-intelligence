# Local general-object tracking baseline

This command runs the same `GeneralObjectTracker` used by serving, with one local
YOLO inference per sampled frame and `sv.Detections.from_ultralytics`. It needs no
embedding model, enrollment, network access, or object-watch library.

## Mac setup and runs

```bash
PYTHON="/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python"
cd "/Users/macbook/Desktop/Career/CV Threat Intelligence/argus-object-watchlists"

# Built-in clip, first 100 processed frames, annotated output (MPS).
"$PYTHON" -m cvti.cli.track_objects \
  --source data/test_clips/normal_street_01.mp4 \
  --weights models/yolov8n.pt --device mps \
  --output-dir runs/object-tracking-normal-100 \
  --max-frames 100 --save-video

# Your own offline video. Pick a new output directory for every run.
"$PYTHON" -m cvti.cli.track_objects \
  --source "/absolute/path/to/video.mp4" \
  --weights models/yolov8n.pt --device mps \
  --output-dir runs/object-tracking-own-video --every-n-frames 2

# Webcam 0 with a window (press Q to stop).
"$PYTHON" -m cvti.cli.track_objects \
  --source 0 --weights models/yolov8n.pt --device mps \
  --output-dir runs/object-tracking-webcam --show
```

Weights must already be a file. The command never downloads weights and refuses
an existing output directory rather than replacing artifacts. If MPS is
unavailable or unsupported, repeat with `--device cpu`. `--max-frames` limits
**processed sampled frames**, not all source frames read; `--every-n-frames 2`
therefore skips every other source frame.

## Inspect artifacts

```bash
less runs/object-tracking-normal-100/summary.json
less runs/object-tracking-normal-100/snapshots.jsonl
open runs/object-tracking-normal-100/annotated.mp4
```

`snapshots.jsonl` is flushed after each processed frame and ends with an explicit
`source_end` record (`eof`, `max_frames`, `user_quit`, `capture_failure`, `error`,
or `interrupted`). An opened file that returns no frames is an error, and a webcam
read failure is never labelled normal EOF. EOF does not reset the tracker and is
not evidence that an object was removed. On an inference/capture error or Ctrl-C,
partial JSONL/video evidence is retained and `summary.json` reports `error` or
`interrupted`; a failed command exits nonzero.

Track trajectories are only observed evidence exported by `GeneralObjectTracker`.
The total `tracks_created` is derived from that wrapper's monotonic sequential
public counter; the CLI does not retain a set of historical IDs. Timing p50/p95
is calculated over a bounded window of the most recent 2,048 processed frames.
Each timing object reports `sample_count`, `observations_total`, `window`, and
`window_capacity`, so long-run percentile semantics are explicit. The summary
also records processed/skipped counts, media metadata, sampling, weight SHA-256,
dependency versions, and wall throughput. Measurements from an unlabelled clip
are operational measurements, not accuracy claims.

For files, tracker time uses monotonic `CAP_PROP_POS_MSEC` media timestamps and
falls back to frame-index/FPS when the media position is invalid or regresses.
Webcams use monotonic elapsed wall time. `annotated.mp4` is always a constant-rate
visualization encoded at reported-FPS/sampling-stride; it is **not time faithful**.
In particular, webcam output is synchronized to processing rather than capture
time, and sampled file output omits skipped intervals/frames. The summary lists
the video artifact only if its writer was successfully created.

The baseline detector uses COCO categories. Start with clearly visible common
objects such as a **bottle, cup, book, backpack, or chair**. It cannot promise an
arbitrary carton or distinguish two visually identical instances reliably.
Identity is local to one tracker session: cross-camera identity is unsupported.
`lost` means temporarily unobserved and `ended` means the track timed out; neither
means physical removal.

## Opt in to serving

The serving site configuration places the booleans directly on each camera (they
both default to `false`):

```json
{
  "cameras": [
    {
      "id": "front",
      "source": "data/test_clips/normal_street_01.mp4",
      "config": "configs/retail_v1.json",
      "general_object_tracking": true,
      "general_object_tracking_overlays": true
    }
  ]
}
```

This local baseline does not silently change the current object-watch/watchlist
path or its deferred bugs. Future identification could optionally add CLIP or
SigLIP embeddings. DINOv2 provides visual features; Grounding DINO provides
open-vocabulary proposals—these are distinct roles and neither is part of this
command.
