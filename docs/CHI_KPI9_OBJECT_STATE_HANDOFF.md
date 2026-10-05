# CHI KPI 9: Objects Left Behind or Removed

## Scope

Opt-in backend implementation on `feat/chi-object-state-events`. It monitors
one object position per named zone using generic detector classes, not SigLIP
product identity. Existing reference-photo recognition remains unchanged.
This is a pilot foundation, not validated inventory accounting or theft detection.

## Processing

The shared YOLO detections feed `ObjectZoneMonitor` in
`cvti/detector/object_state.py`. A stable observed object followed by sustained
absence and changed pixels creates a removal candidate. A previously observed
empty position followed by a stationary object creates a left-behind candidate.
Startup stock is not treated as newly abandoned.

Person overlap suspends decisions; long occlusion, inference failure, source
reset, frame gaps, changed geometry and broad scene changes discard history.
Ambiguous multiple targets reset that position. Missed labels without visual
change cannot establish removal. These are conservative heuristics, not proof
that every possible occlusion or lighting change is handled.

Customization rules select candidates. A labelled BEFORE/AFTER montage is
copied into immutable verification evidence, then sent through the existing
VLM gate and alert pipeline. The prompt explicitly separates removal from theft.
Mock verification is only a plumbing test, never an accuracy test.

## Enable for a Test Camera

Use an existing zone file drawn for that camera. Add these fields to its entry
in the site configuration, preserving its source and other settings:

```json
{
  "config": "configs/chi_object_state_v1.json",
  "zones": "configs/your_camera_zones.json",
  "object_state_zones": [
    {
      "zone": "walkway",
      "labels": ["suitcase", "backpack"],
      "mode": "left_behind",
      "dwell_seconds": 10,
      "stable_seconds": 2
    },
    {
      "zone": "storage_position",
      "labels": ["suitcase", "backpack"],
      "mode": "removed",
      "absent_seconds": 5,
      "stable_seconds": 2
    }
  ]
}
```

The zone file must contain those exact names. Keep each zone tight around one
position. The supplied rules match those names; adjust both policies and rules
when renaming. To retain other custom rules, merge these two rules into the
camera's existing rules configuration instead of replacing its config path.

Use a stationary camera and at least a few successful samples per second.
The default maximum gap is 2.5 seconds and minimum observations is three.
Ten-second placement dwell above is for testing, not a recommended site policy.
Absent policies leave existing cameras unchanged. There is no new UI policy
editor in this change: geometry uses existing zones; these policies use JSON.

Standard COCO YOLO does not detect a generic cardboard-box class. Adding
`"carton"` to this configuration does not teach the model to recognize cartons.
CHI product/carton deployment requires a detector/proposal path that actually
supplies those boxes; reference matching remains a separate integration.

## Manual Acceptance

1. Start with an empty walkway, wait for baseline, place a supported item and
   step away. Expect one candidate after the dwell, subject to VLM verification.
2. Start with an item in storage, wait for a stable baseline, remove it and
   step away. Expect one removal candidate after sustained visible absence.
3. Leave stock present at startup: no left-behind alert.
4. Walk across an item without removing it: no removal alert.
5. Cover/uncover the camera, reconnect it, and change zone geometry: old
   history must not trigger a removal alert.
6. Inspect saved evidence and actual gate verdicts, including rejected cases.
   Record expected versus observed events for positive and negative clips.

Automated tests use synthetic images and exercise rules and the verification
queue. Real camera, real VLM accuracy, CHI cartons, crowded storage and packaged
desktop deployment still require acceptance testing. No accuracy claim is made.
