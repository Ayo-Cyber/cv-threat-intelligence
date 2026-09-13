# Chi Pilot Testing

## Scope And Current Status

This document currently covers only Chi pilot scenario 10: a shopper pockets
merchandise or places it into a personal bag. Motion scenarios 4 and 5 will be
added later.

The repaired architecture has automated regression coverage, but the changed
TrueSight prompt is **unmeasured**. The frozen golden corpus is unavailable, so
`docs/prompt_baseline.json` records zero scored cases and no precision or recall.
The matrix below is the required recording and acceptance protocol, not a claim
that the seven Chi cases have already passed.

## Repaired End-To-End Path

```text
camera frame
  -> shared YOLO detections (people and COCO personal bags)
  -> per-camera pose inference on heavy_stride samples
  -> per-person bag assignment and temporal ConcealmentDetector
  -> concealment RawEvent with destination, score, components, reasons,
     limited-evidence flag, track ID, subject box, and associated bag box
  -> CustomizationEngine plus reviewed Agent Mapper scene compatibility
  -> deduplicated shared alert queue
  -> TrueSight with 3 chronological full frames plus the subject crop
  -> confirmed event persistence, notification, and UI evidence
```

Skipped pose frames do not erase temporal history. `expire(timestamp)` still
runs every video frame and removes stale tracks after the grace period. Personal
bags are limited to COCO backpack, handbag, and suitcase classes and are
assigned to one nearby pose track; an exact ownership tie is left unassigned.
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
| Candidate | first candidate timestamp or `none`, destination, peak score, components, limited flag, associated bag, track ID |
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

## Exact Regression Commands

Run from the repository worktree root. This worktree intentionally uses the
shared project virtual environment at the absolute path below.

Prompt fingerprint/status check:

```bash
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" tools/prompt_regression.py check
```

Focused concealment regression:

```bash
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" -m pytest tests/test_concealment.py tests/test_heavy_models_earn_their_frames.py tests/test_serving.py tests/test_prompt_regression.py tests/test_gate_evidence_quality.py -q
```

Full Python regression:

```bash
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" -m pytest
```

These commands validate the implementation contracts. They do not replace the
seven-clip Chi run or measure the changed prompt.

## Prompt Measurement When The Corpus Returns

First verify that the restored golden directory contains its manifest, cases,
and frames. Start Ollama with the pilot model, then run a complete fresh replay:

```bash
MPLCONFIGDIR=/private/tmp "/Users/macbook/Desktop/Career/CV Threat Intelligence/cv-threat-intelligence/.venv/bin/python" tools/prompt_regression.py run --golden-dir runs/eval/golden --gate-provider ollama --gate-model gemma3:4b --sensitivity balanced --fresh --verbose
```

Only a complete replay with zero errored cases is a measurement. Review the
metrics and model digest before deliberately rerunning with
`--update-baseline`; partial or errored runs must remain non-baseline results.
