# CV Threat Intelligence — Architecture Plan

A three-layer architecture for context-aware, customizable threat detection.

## High-level diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      CAMERA FRAMES                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
   ┌──────────▼─────────┐         ┌─────────▼─────────┐
   │   AGENT MAPPER     │         │  DETECTION CORE   │
   │   (runs once /     │         │  (every frame)    │
   │    periodically)   │         │                   │
   │                    │         │ • RT-DETR         │
   │ VLM (Claude/CLIP)  │         │ • Pose + ByteTrack│
   │ → scene context    │         │ • State machines  │
   └──────────┬─────────┘         └─────────┬─────────┘
              │                             │
              │  scene_context.json         │  raw_events.json
              │                             │
              └──────────┬──────────────────┘
                         │
              ┌──────────▼─────────┐
              │  CUSTOMIZATION     │
              │  ENGINE            │
              │                    │
              │ Apply user rules   │
              │ (user_config.json) │
              └──────────┬─────────┘
                         │ candidate alerts
                         │
              ┌──────────▼─────────┐
              │  VERIFICATION GATE │
              │  (VLM confirms)    │
              │                    │
              │ Claude Vision      │
              └──────────┬─────────┘
                         │
                         ▼
                    ALERT / NO ALERT
```

## Layers

### 1. Agent Mapper

Runs **once per session** or **every ~5 minutes** — not every frame. Sends a sample frame to a VLM (Claude Vision or local) and asks it to describe the environment. The output becomes the static context for that camera/session.

**Purpose:** so the threat detector knows it is looking at a *retail shop*, not a *parking lot*. Context dramatically changes what counts as a threat.

### 2. Detection Core

Existing pipeline in `detector.py`:

- RT-DETR (object detection)
- YOLOv8n-pose (skeletons)
- ByteTrack (person IDs)
- `assess_threat` / `assess_violence` / `TheftDetector` state machines
- `ViolenceTemporalGate` rolling-window confirmation

Outputs per-frame raw events (no filtering, no user rules).

### 3. Customization Engine

Reads `user_config.json` from the frontend. Each user defines their own threat rules without code changes. Engine matches `raw_events` against the rules and produces candidate alerts.

### 4. Verification Gate

Only fires when a rule matches. Sends the trigger frame + scene context + candidate alert to a VLM (Claude Vision). VLM confirms or rejects with a reason. This is the structural fix for false positives identified in Week 1 / Week 2 eval (weapon model over-fires on shop items, violence heuristic fires on crowded scenes).

## Data contracts (JSON)

### scene_context.json — Agent Mapper output

```json
{
  "environment_type": "retail_shop",
  "scene_description": "Small clothing boutique with 2 visible aisles and a checkout counter on the right",
  "zones": [
    {"id": "checkout", "bbox": [800, 100, 1200, 400], "role": "safe"},
    {"id": "aisle_1",  "bbox": [100, 200, 700, 800],  "role": "merchandise"},
    {"id": "exit",     "bbox": [1300, 600, 1500, 900], "role": "transition"}
  ],
  "expected_actors": ["shoppers", "staff"],
  "watch_for": ["concealment", "loitering_near_exit"],
  "captured_at": "2026-05-16T10:00:00Z"
}
```

### user_config.json — frontend → backend

```json
{
  "use_case_id": "retail_v1",
  "rules": [
    {
      "name": "shoplifting",
      "trigger": {"detector": "theft", "state": "DEPART"},
      "context_filter": "person.zone != 'safe'",
      "priority": "high"
    },
    {
      "name": "after_hours_intrusion",
      "trigger": {"detector": "person", "count": ">0"},
      "time_filter": "22:00-06:00",
      "priority": "critical"
    },
    {
      "name": "loitering_at_atm",
      "trigger": {"detector": "person", "dwell_seconds": ">60"},
      "context_filter": "person.zone == 'atm'",
      "priority": "medium"
    }
  ]
}
```

### alert.json — verification request to VLM

```json
{
  "frame": "<base64-encoded image>",
  "scene_context": {
    "environment_type": "retail_shop",
    "scene_description": "..."
  },
  "candidate_alert": {
    "rule_name": "shoplifting",
    "detected_state": "DEPART",
    "person_id": 3,
    "object_label": "handbag"
  },
  "question": "Does this frame confirm shoplifting given the scene is a clothing shop?"
}
```

### alert.response.json — VLM verification response

```json
{
  "confirmed": true,
  "confidence": 0.87,
  "reason": "Person placed item directly into their jacket and turned toward exit without visiting checkout.",
  "alert_priority": "high",
  "timestamp": "2026-05-16T10:14:23Z"
}
```

## Why this architecture works

| Problem (observed) | Layer that solves it |
|---|---|
| Pipeline doesn't know it's a shop vs street → fires on everything | **Agent Mapper** — context one-shot |
| Different customers need different threat definitions | **Customization Engine** — JSON rules, no code changes |
| Detector over-fires (Week 1 FPR ~0.6 on shop items, Week 2 FPR 0.67 on crowds) | **Verification Gate** — VLM kills FPs structurally |
| Theft vs legitimate shopping ambiguity | **Gate + scene context** — VLM understands intent |

## Recommended build order

1. **Schemas first (1 day)** — write `schemas/scene_context.schema.json`, `schemas/user_config.schema.json`, `schemas/alert.schema.json`. Gives frontend and backend a contract to build against in parallel.

2. **Verification Gate (1–2 days)** — easiest win, biggest impact. Wrap Claude Vision API. Call only when current state machines fire. This alone should slash FPR on the existing Week 2 baseline.

3. **Customization Engine (2–3 days)** — small rules evaluator reading `user_config.json` against `raw_events`. Start with a minimal expression language (`==`, `!=`, `>`, `<`, AND/OR).

4. **Agent Mapper (last)** — needs zone annotation infra. Highest-effort piece. Defer until 1–3 are working.

## Notes

- The verification gate should be called **only when the state machine fires**, not every frame. Cost and latency would be prohibitive otherwise. ~$0.001 per alert at Claude pricing is acceptable; ~$0.001 per frame is not.
- Agent Mapper runs infrequently because scenes don't change rapidly. A camera looking at the same shop doesn't need re-mapping every frame.
- The Customization Engine is the unlock for productization — same backend, many use cases. This is what makes the product sellable to different verticals (retail, banking, schools, malls).
