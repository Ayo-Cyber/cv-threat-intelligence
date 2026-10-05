# Concealment Reliability: Diagnosis and Validation

## Scope

Latest live checkpoint: final Electron run with `--wig` saved to
`runs/chi_validation/scenarios/10_concealment/evidence/2026-10-03T21-38-27.341Z/`.
Ollama gemma3:4b: 3 verified, 2 confirmed, 1 rejected, 0 errors, 0 pending,
2 deduped; median verification latency 22.37 seconds. These are model outcomes,
not measured accuracy. One confirmation says a mannequin was placed into a
pocket, so object-level hallucination remains. Do not represent this as a
production-ready or fully validated concealment detector.

This is a gesture candidate generator plus visual verification, not proof of
theft, non-payment, ownership or criminal intent. The user reports concealment
in both retail clips; absence of alerts is not evidence of normal behaviour.
The footage and its captions are not independently labelled ground truth.

## Problems Found and Changes

- Frame-count dwell and persistence changed behaviour with sampled FPS. Dwell
  now uses observed elapsed time; legacy frame parameters are interpreted at
  the original 10Hz calibration. Gaps over .55s do not contribute dwell.
- Reach evidence aged out too quickly at low pose cadence. The gesture window
  is now 2.4s. Persistence requires two observations and elapsed duration.
- Hand resting near the waist could exceed the score threshold. Candidates now
  also require a retraction component of at least .25. This is still a heuristic,
  not evidence of a product; walking arm swings may still propose candidates.
- Sustained scores generated repeated proposals. A four-second per-track
  candidate cooldown bounds emission, independent of downstream queue dedup.
- Video loops rewound timestamps without clearing gesture history. Rollback
  now clears concealment/ownership state and camera evidence/pose history.
  Duplicate timestamps cannot contribute repeated observations.
- Pose IDs were confused with ByteTrack IDs. The pose subject box is now carried
  into evidence metadata; no unrelated largest-person fallback is used.
- Ten recent frames were not a stable evidence duration. Concealment now samples
  three chronological images from up to four seconds of JPEG replay history,
  followed by the current subject crop. Replay images are capped at 640px width;
  this trades memory for resolution, and memory-guard trimming can shorten it.
- Generic threat prompts encouraged ambiguous confirmations. Concealment has a
  separate prompt requiring an item-to-pocket/clothing/personal-bag sequence and
  ignoring news captions. The runtime additionally requires item_visible=true,
  action=insertion and an allowed destination. Other detectors retain their
  prompt behavior; transport errors retain existing fail-visible handling.

These checks cannot prevent a model from hallucinating all required fields.
Unknown/occluded actions are not independently established negatives. Human
review remains necessary. No claim of universal or production acceptance.

## Evidence So Far

All files below are relative to `runs/chi_validation/scenarios/10_concealment/`.

- `trace/original-gate/`: original early-walking VLM inputs, preserved.
- `pose-probe-timed-4fps.json`: unchanged .63 score threshold, 4fps requested,
  pose every sample. Wig clip candidates at18.4s and26.67s; second clip at4.67s,
  7.47s and8.88s. Candidate timing is not an accuracy metric.
- `verified-timed-4fps.json`: isolated first-candidate verification showed the
  prompt alone still confirmed while saying no clear insertion occurred.
- `verified-structured-4fps.json` and matching `-gate/`: structured-contract
  verification returned a pocketing verdict for the wig clip at18.4s. Second
  clip also confirmed, but inspected imagery raises actor-mixing concerns:
  a background bag-carrier and a foreground person are both visible. Do not
  count this as validated true-positive performance.
- `evidence/2026-10-03T21-31-40.735Z/`: two-camera real Electron run with timed
  scoring, BEFORE the structured contract. Five verified, three confirmed, two
  rejected, twelve pending at snapshot. Several verdicts remained unsupported.
  Memory guard trimmed history. Cumulative monitor.log contains earlier runs.

111 focused tests passed, including temporal behavior at2/4/8/15/30fps, long
hand-rest negatives, rewind/duplicate handling, subject ID collision, evidence
ordering, gate contract and fail-visible regressions. Synthetic tests do not
replace independent retail-video evaluation.

## Reproduction

Use the configured project Python to run `scripts/prepare_kpi10_test.py`.
Then from `Frontend` run:

```sh
node scripts/chi-kpi10.mjs --timed
node scripts/chi-kpi10.mjs --wig
```

Run separately, not concurrently. Each starts the actual desktop application
and local engine, runs120s, captures screenshots/config/health/gate artifacts,
and stops. The second isolates the wig clip to distinguish model behavior from
two-camera resource contention. User model and test assets remain local/ignored.

## Approach Comparison and Next Acceptance Work

[Veesion's technology description](https://veesion.io/en/about/our-technology/)
describes trained video deep learning for gestures. Its
[solution page](https://veesion.io/en/our-solution/) describes configurable gesture
types and review of video alerts. It does not publish enough implementation
detail to claim our distance heuristic reproduces its model or performance.

[MMAction2](https://github.com/open-mmlab/mmaction2) supports both RGB video models
such as VideoMAE and skeleton models such as PoseC3D. These are candidate
frameworks, not ready-made accurate shoplifting models. Our repository already
has a VideoMAE serving integration, but its expected checkpoint directory
`runs/video_finetune/videomae` is absent in this checkout. Reuse that integration
and evaluate a properly licensed retail-trained checkpoint before adding another
stack. Generic action weights alone do not solve shoplifting recognition.

Recommended next validation: independently annotate source action start/end,
actor, item, destination and visibility. Include phones, clothing adjustment,
open carrying, shelf returns, carts, customer shopping bags, occlusion and
multiple nearby actors. Hold out cameras/stores, not just adjacent frames.
Measure event recall, false alerts per camera-hour, actor attribution, duplicate
alerts, verification latency and queue backlog on deployment hardware.

Likely next engineering layer: per-hand temporal proposals plus subject-centred
RGB clip classification, keeping a full scene view for context. Train/validate
on retail actions and difficult negatives; do not replace the current model with
an unvalidated generic checkpoint or silently accept all gestures as theft.
