# KPI 10 Three-Camera Checkpoint

This is a bounded diagnostic run, not acceptance testing or proof of theft.

## Sources

1. `data/test_clips/theft_shop_01.mp4`: existing wig-shop news clip. User reports
   concealment; event-level labels have not been independently annotated.
2. `data/test_clips/theft_shop_02.mp4`: existing multi-person retail clip. User
   reports concealment; actor association and occlusion remain open risks.
3. `runs/chi_validation/caviar/WalkByShop1front_excerpt.mp4`: newly downloaded
   quiet-shop control from the official CAVIAR archive:
   https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA2/WalkByShop1front/WalkByShop1front.mpg
   Dataset index: https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/
   Publisher describes walking/browsing and entering/exiting shops. Treat as a
   candidate negative control, not independently labelled frame-level truth.
   Download timed out after180s at3,215,941bytes. Decoded the first500frames
   (20s,25fps,384x288) into a standalone MP4; this is NOT the complete source.
   Sampled frames show a quiet storefront/displays, not active handling of goods.
   This does not substitute for a normal-shopping/clothing-adjustment hard negative.
   Provenance metadata and contact sheet sit beside the excerpt.
   Retain locally for evaluation; redistribution/commercial packaging rights
   have not been established. Do not bundle these videos in customer releases.

## Reproduce

From the repository root, using the existing Python environment:

```sh
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" scripts/prepare_kpi10_test.py
cd Frontend
node scripts/chi-kpi10.mjs --three
```

Actual Electron app, local YOLO pose pipeline and Ollama gemma3:4b, three
simultaneous feeds, requested4fps/512px and heavy_stride1, 180 seconds. No new
threshold calibration for the new video. Person boxes are enabled. Captures
overlays, screenshots, health, gate artifacts and config. App closes afterward.
Shared database contains older incidents: count only rows created after
`run-start.json` in this run's evidence directory. Repeated file loops are not
independent samples. Pending work at the end is not a negative detection result.

## Next KPI

## Observed Results (2026-10-04)

Evidence: `runs/chi_validation/scenarios/10_concealment/evidence/2026-10-04T10-45-49.825Z/`.
Per-camera report: `checkpoint-summary.json`. Actual screenshots show all3 feeds.

| Camera | Candidate audit rows | Completed outcomes at snapshot | New incidents |
| --- | ---: | --- | ---: |
| retail_1 | 8 | 4 admitted without verdict; 4 deduplicated | 0 |
| retail_2 | 14 | 7 rejected; 1 needs review; 6 admitted without verdict | 1 inconclusive |
| normal_shopping (quiet-shop excerpt) | 0 | No candidates | 0 |

Engine:1734 frames, all cameras connected, each ingest24fps, no stale drops.
Gate:8 completed,0 confirmed,1 review_required,7 rejected,0 errors,9 pending;
median19.46s per verification, which excludes queue waiting. Ten audit rows
without verdict can include an in-flight request plus9 queued. The run stopped
at its bounded cutoff; unprocessed candidates are NOT negatives. Repeated source
loops are not independent samples. Existing incidents visible in the sidebar
are historical; only one new incident belongs to this measured window.

Disposition: partial checkpoint, NOT KPI10 acceptance. Leave temporal evidence,
visual accuracy and multi-camera queue fairness/throughput as explicit backlog.
No further tuning in this checkpoint at the user's request to move on.

## KPI 9 Preparation

Proceed to KPI 9 (object left behind/removed). Existing backend implementation
and `runs/chi_validation/caviar/LeftBag_PickedUp.mpg` are available. First review
that clip, define an object zone, and test placement/removal plus an occlusion
negative. KPI 3 product identity remains a separate SigLIP/model-install workstream.
Prepared and inspected a timestamped source review sheet at
`runs/chi_validation/scenarios/09_object_state/left_bag_source_review.jpg`.
Source duration54.48s. Placement/pickup is near the upper wall/display area;
zone calibration and real detector coverage must be checked before claiming a
live pass. Preparation script: `scripts/prepare_kpi9_review.py`.
