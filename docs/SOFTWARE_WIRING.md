# Software Wiring

This project has two runnable pipelines today:

- `detector.py`: general threat-detection demo for weapons, violence, theft states, and classifier output.
- `retail_pipeline.py`: retail-focused orchestration for zones, concealment candidates, user rules, and VLM verification.

Both pipelines should communicate through the same small set of shared contracts.

## Module Ownership

| Module | Owns | Should not own |
|---|---|---|
| `cvti/contracts.py` | Shared dataclasses passed between layers: `RawEvent`, `CandidateAlert`, `VerificationResult` | CV/model code, rule evaluation, API calls |
| `cvti/event_adapters.py` | Converting detector, zone, and concealment state into `RawEvent`s | Rule policy, VLM prompts, frame processing |
| `cvti/rules/customization.py` | Loading user rules and evaluating `RawEvent`s into `CandidateAlert`s | Detector-specific conversion logic |
| `cvti/verification/gate.py` | Confirming/rejecting `CandidateAlert`s with mock or VLM providers | Detection, tracking, rule matching |
| `cvti/retail/zones.py` | Zone geometry, membership, dwell accounting, annotation | Theft decisions |
| `cvti/retail/concealment.py` | Pose-sequence concealment candidate scoring | Final alert decisions |
| `cvti/pipelines/retail_pipeline.py` | Retail orchestration loop | Shared contracts or rule internals |
| `detector.py` | General demo orchestration loop and legacy threat heuristics | Shared contracts or product-specific rule policy |

## Intended Flow

```text
model / tracker output
  -> pipeline-specific assessment objects
  -> event_adapters.py
  -> RawEvent[]
  -> customization.py
  -> CandidateAlert[]
  -> verification_gate.py
  -> VerificationResult
  -> evidence / alert output
```

## Refactor Rule

If a piece of code needs to be used by both `detector.py` and `retail_pipeline.py`, it should usually live outside both scripts. Prefer:

1. `cvti/contracts.py` for neutral data shapes.
2. `cvti/event_adapters.py` for conversion into events.
3. A focused module named for the responsibility, not for the pipeline.

Keep the CLI scripts as thin orchestration layers over time.
