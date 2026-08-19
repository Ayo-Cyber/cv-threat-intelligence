# Agent Mapper Plan

## Purpose

The Agent Mapper is an infrequent scene-understanding layer that sits before the threat pipeline. Its job is **not** to detect threats frame by frame, and it is **not** to recommend threat categories or presets. Its job is to describe the environment so the rest of the system can interpret detections correctly.

Core questions it should answer:
- What kind of place is this camera looking at?
- What actors are normally expected here?
- What important structural zones are visible, if any can be suggested confidently?

What the Agent Mapper deliberately does **not** do (2026-05-24 refactor):
- it does **not** emit `risk_hints`
- it does **not** emit `suggested_preset`
- it does **not** speculate about threats, danger, or intent

Threat policy is the job of the Customization Engine reading the user's authoritative `user_config.json`. Putting threat semantics in the Mapper would tightly couple it to the GTM rule library, introduce VLM speculation noise into the most upstream layer, and pre-empt the user's authority over their own threat definition.

Output: `scene_context.json` (descriptive only)

## MVP Scope

### What v1 should do

1. Accept a source:
   - webcam
   - RTSP stream
   - local video file
   - local image file
2. Capture one or a few representative frames.
3. Send the best frame to a VLM with a strict JSON-only prompt.
4. Parse and validate the response against a schema.
5. Save a `scene_context.json` artifact to disk.

### What v1 should not do

- run every frame
- decide whether a threat happened
- fully replace human setup
- auto-enforce rules directly
- generate permanent zones without human review

## Why This Matters

The same raw event means different things in different places:
- loitering at an estate gate after midnight is suspicious
- loitering in a mall corridor during business hours may be normal
- a person near a generator during a blackout in Nigeria may be high risk

The Agent Mapper gives the system this missing environmental context.

## Proposed File Layout

```text
docs/
  AGENT_MAPPER_PLAN.md
prompts/
  agent_mapper_prompt.txt
schemas/
  scene_context.schema.json
agent_mapper.py
runs/
  context/
    <camera_or_source_id>/
      source_frame.jpg
      scene_context.json
```

## Allowed Vocabulary

### Environment Types

These should stay bounded so downstream logic is stable.

- `estate_gate`
- `estate_street`
- `perimeter_fence`
- `retail_shop`
- `mall_corridor`
- `office_lobby`
- `office_floor`
- `parking_lot`
- `banking_hall`
- `atm_area`
- `warehouse_floor`
- `generator_area`
- `residential_interior`
- `residential_exterior`
- `unknown`

### Zone Roles

- `entry`
- `exit`
- `transition`
- `safe`
- `restricted`
- `merchandise`
- `checkout`
- `parking`
- `perimeter`
- `asset`
- `unknown`

## Scene Context Schema

The exact JSON schema is stored in:
- `schemas/scene_context.schema.json`

Expected output shape:

```json
{
  "camera_id": "estate_gate_cam_1",
  "source_type": "video_file",
  "environment_type": "estate_gate",
  "scene_description": "Main estate gate with a vehicle entrance, pedestrian gate, and security post.",
  "expected_actors": ["residents", "security_guards", "visitors", "vehicles"],
  "zones": [
    {
      "id": "main_gate",
      "label": "main_gate",
      "role": "entry",
      "bbox": [820, 110, 1240, 470]
    }
  ],
  "confidence": 0.86,
  "generated_at": "2026-05-16T17:40:00Z",
  "source_frame_path": "runs/context/estate_gate_cam_1/source_frame.jpg",
  "notes": "Human should verify zone boundaries before activation."
}
```

## `agent_mapper.py` Structure

Recommended module structure:

### 1. `parse_args()`

Support:
- `--source`
- `--camera-id`
- `--sample-count`
- `--sample-interval-seconds`
- `--output-dir`
- `--provider`
- `--model`
- `--prompt-file`
- `--save-frame`

### 2. `normalize_source()`

Match current detector conventions:
- numeric webcam index stays numeric
- RTSP / file / image stays string

### 3. `capture_sample_frames()`

Input:
- source
- sample count
- interval

Output:
- list of sampled frames + timestamps

Rules:
- image input: one frame only
- video file: sample evenly
- webcam / RTSP: sample across a short window

### 4. `choose_representative_frame()`

Pick the best candidate frame using simple heuristics:
- highest brightness within acceptable range
- lowest blur estimate
- avoid mostly-black frames

This reduces bad VLM outputs caused by poor sample quality.

### 5. `build_prompt()`

Load `prompts/agent_mapper_prompt.txt` and inject:
- camera id
- source type
- allowed enums

### 6. `call_vlm()`

Thin provider wrapper.

Responsibilities:
- send image + prompt
- request strict JSON output
- return raw text

Keep this isolated so providers can change later.

### 7. `parse_and_validate_scene_context()`

Responsibilities:
- parse JSON
- validate against `schemas/scene_context.schema.json`
- apply fallback defaults where needed

If validation fails:
- log the raw response
- fall back to `environment_type = "unknown"`

### 8. `save_scene_context()`

Write:
- chosen frame to `source_frame.jpg`
- validated JSON to `scene_context.json`

### 9. `main()`

Flow:
1. parse args
2. sample frames
3. choose representative frame
4. call VLM
5. validate response
6. save outputs
7. print saved paths

## Prompt Template

The prompt file lives at:
- `prompts/agent_mapper_prompt.txt`

Design goals:
- bounded vocabulary
- JSON only
- no freeform prose outside JSON
- output should be stable enough for downstream parsing

## Integration With Current Pipeline

### Near-term

Before a full customization engine exists, Agent Mapper can still be used to:
- provide descriptive scene context for downstream layers
- act as the input to a deterministic Preset Recommender (separate component, no VLM) that maps `environment_type` to a preset and a default rule pack
- provide context to a future verification gate
- guide manual operator setup

### Later

Once the customization engine exists, `scene_context.json` can feed:
- zone-aware filtering
- verification prompts (give the VLM scene context when confirming alerts)
- the Preset Recommender, whose suggested rule pack the user then accepts or edits to produce the authoritative `user_config.json`

## Testing Strategy

### Yes: online video testing is compatible and should remain a first-class workflow

This is important.

Your collaborator has been using online videos in sync meetings. The Agent Mapper should absolutely support that workflow, because:
- it is fast
- it gives diverse scenes
- it is good for early architecture testing
- it avoids waiting for live camera access

### Recommended testing inputs for v1

1. **Local image files**
   - fastest unit-style tests
   - useful for prompt iteration

2. **Local video clips**
   - should be the main test path initially
   - includes:
     - downloaded online clips
     - curated test clips in repo
     - short manually recorded videos

3. **RTSP streams**
   - validate real deployment compatibility after local clips work

4. **Webcam**
   - useful for smoke tests, but not the best primary path for the Agent Mapper

### Recommended first test set

Build a small folder of representative scenes:
- estate gate
- estate street
- retail shop interior
- office lobby
- parking lot
- warehouse floor
- generator area
- mall corridor

For each clip, the expected test output should be manually defined:
- expected environment type
- acceptable preset
- acceptable risk hints

### What to measure

For Agent Mapper v1, do not overcomplicate the metrics.
Track:
- correct environment classification
- scene_description quality (manual eyeball pass)
- valid JSON generation rate
- average runtime per mapping call
- whether the model leaks threat language despite the prompt forbidding it

### Suggested testing workflow

1. Use downloaded online clips first.
2. Run Agent Mapper on each clip.
3. Inspect:
   - chosen sample frame
   - generated `scene_context.json`
4. Compare output with expected scene type and preset.
5. Refine prompt and enums until outputs stabilize.

### Key compatibility conclusion

Yes, the current online-video workflow is still fully compatible with the Agent Mapper approach.

In fact, online videos are probably the best initial testing path for Agent Mapper because the feature is about **scene understanding**, not just live-camera runtime.

## Recommended Implementation Order

1. Create and lock the `scene_context` schema.
2. Create the prompt template.
3. Build `agent_mapper.py` with file/video/image support first.
4. Test on online videos and local clips.
5. Add webcam / RTSP verification after local clip behavior is stable.
6. Only after that, integrate it with preset selection and the future verification gate.

## Immediate Deliverables

The next engineering deliverables should be:
- `schemas/scene_context.schema.json`
- `prompts/agent_mapper_prompt.txt`
- `agent_mapper.py` skeleton
- a small clip-based test set

## Summary

The correct v1 Agent Mapper (post 2026-05-24 refactor) is:
- a scene classifier
- a scene describer
- a zone suggester
- a JSON artifact producer

It is deliberately **not** a preset recommender or risk-hint generator. Those responsibilities live in a separate deterministic Preset Recommender component and ultimately in the user's authoritative `user_config.json`.

It should be tested first on downloaded and local video clips, because that is already compatible with how the team has been working and is the fastest path to reliable iteration.
