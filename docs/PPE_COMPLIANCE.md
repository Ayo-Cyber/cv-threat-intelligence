# PPE compliance — per person, three states, the customer's policy

*16 Sep 2026. Package: `cvti/ppe/`. Tests: `tests/test_ppe_compliance.py`.*

## The claim we make, and the ones we refuse to

A person on a camera with a `ppe` policy is one of:

| Verdict | Meaning |
|---|---|
| **violation** | a required item is *confidently absent* — the body region was visible, checked N times, never carrying it |
| **compliant** | every required item is confidently present **and on the right body region** |
| **unable** | something required could not be assessed: feet below the frame, two people overlapping so a vest's owner is unclear, person too small, detector down |
| not_required | the zone they stand in requires nothing |

There is no percentage. Two of three items is not "67 % compliant" and it is
not a violation either — it is *unable* until the third is confidently
absent, then a violation. Failure to detect is never proof of absence; a
visible region checked repeatedly with nothing there is.

What CCTV cannot establish is carried on the item (`visual_only_note`) and
rides in every alert payload: boots are *appearance only — cannot establish
steel toe, puncture resistance or certification*.

## Who decides what is required

The customer, in configuration — per zone. The model never does.

```json
"ppe": {
  "required": ["helmet"],                       // anywhere on this camera
  "zones": {
    "loading_bay": ["helmet", "vest", "boots"], // union across zones a person is in
    "office_corridor": []                       // explicitly nothing here
  },
  "items": { "blue_uniform": {"region": "torso", "phrases": ["blue work uniform"]} },
  "cadence_seconds": 1.0, "confirm_observations": 3, "window_seconds": 8.0,
  "min_person_height_px": 80, "clear_after_seconds": 30.0, "priority": "high"
}
```

Zones are the camera's drawn zones (the same `zones` file the loitering and
entry/exit rules use). Catalog items are industrial, not construction-only:
`helmet hairnet goggles mask vest apron coverall gloves boots`, each bound to
the body region it is worn on. Unknown item names are rejected at load —
a typo must not become a requirement nobody can satisfy.

## The pipeline

```
tracker person boxes + zone membership   (already computed by the camera loop)
        ↓
item detector on the frame               (open-vocab today; a trained PPE model drops in)
        ↓
associate: item centre inside THIS person's head/torso/hands/feet band
        ↓ one owner → theirs · two owners → unknown for both · none → ignored (helmet in a hand)
visibility: region inside the frame? person tall enough? feet not clipped?
        ↓
per-track evidence, timestamped, expiring (window 8 s)
        ↓ absent needs N unanimous checks · present needs a majority · else unknown
zone policy → violation / compliant / unable
        ↓
ONE alert per person per missing-set; coverage counters → ppe_status.json
```

`cvti/ppe/assess.py` holds every step but the detector. Regions are bands of
the person box (head 0–22 %, torso 18–62 %, hands 38–78 %, feet 84–103 %) —
generous on purpose; pose estimation can replace them if it earns its cost.

## Where it runs, and why the streams don't slow down

`cvti/ppe/scanner.py` is one thread for the site, off the camera path. It
peeks the frame the engine already decoded and the boxes/zones the tracker
already produced — the camera loop never waits for it. One inference in
flight at a time, per-camera cadence (default 1 Hz); a cycle that overruns
skips to the newest frame and counts `dropped_cycles` rather than judging a
queue of stale frames. It does not use the VLM.

`ppe_status.json` (next to `english_rules_status.json`) reports, per camera:
people seen · assessed · compliant · violations · **unable and why** ·
unable rate · alerts · dropped cycles · cycle ms · detector health ·
`degraded`. A quiet board with a high unable rate is a coverage problem, not
a compliant site — the status makes that visible.

## Measured (16 Sep, zero-shot YOLO-World `yolov8s-worldv2`)

| Item | Clip | What we saw |
|---|---|---|
| hi-vis vest | `data/ppe_clips/vest_src.mp4` (everyone vested) | best torso score per person 0.05–0.53, median ~0.2 |
| hard hat | `data/ppe_clips/ppe_compliance.mp4` | detections ≤ 0.22 — the clip's people are too small for the person tracker at 0.35 |

On the vest clip the zero-shot model produced **no vest detection at all on
45 of 70 vested torsos** (recall ≈ 35 % at any floor). The assessment engine
behaved as designed (unanimous-absent, ownership, visibility, one alert per
person) — and that is exactly why it produced 11 false "missing vest" alerts:
it trusted the detector's silence. **The zero-shot detector is the weak
link**, and the engine now says so instead of alerting:

1. **Shadow items.** Each catalog item records whether the zero-shot model's
   silence can be trusted (`absent_reliable_zero_shot`). Hard hat: yes
   (0.6–0.8 when worn). Vest, gloves, boots, mask, goggles, hair net: no.
   With the open-vocab detector (`zero_shot = True`) a shadow item is
   assessed and counted but its "not seen" becomes *unknown — zero-shot
   detector cannot confirm absence (shadow)*, never a violation. A seen
   vest is still *present*. `ppe_status.json` lists `shadow_items` per
   camera; a site that accepts the risk opts in with `"trust_absent": ["vest"]`.
2. **The detector seam** is `detect(frame, phrases, floor=) → [{phrase, score, box}]`
   plus a `zero_shot` flag. A trained industrial PPE model (SH17: 17 classes
   incl. hi-vis, hard hat, gloves, face masks, ear protection) wraps into that
   shape with `zero_shot = False`, and every shadow item goes live without
   touching policy, association or evidence.

## What to prove for CHI, in order

1. **Camera suitability** — can the camera see heads, torsos, feet at
   ≥ 80 px person height? The unable-reason counters answer this per camera.
2. **Helmet + vest in one or two zones, shadow mode** — review alerts against
   footage; measure violation precision/recall, false alerts per camera-hour,
   wrong-person associations, unable rate, p95 cycle ms at the intended
   camera count.
3. **Trained item detector** (SH17) once the zero-shot numbers are the
   ceiling — then boots/gloves where the camera can actually see them.
4. Hair net / apron for the production floor (food & beverage), same engine.

## What else in Argus this gives us

* **The alert builder** gets a fourth trigger type — *PPE requirement per
  zone* — that is pure configuration on top of the zone editor.
* **Three-state verdicts** (violation / compliant / unable) and the coverage
  counters are the pattern every other per-person rule should adopt
  (uniform in the production area, badge visible at the gate).
* `latest_zones` — per-track zone membership is now published by the camera
  loop for any off-path scanner, not just PPE.
