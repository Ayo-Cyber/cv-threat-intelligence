"""Argus Engine HTTP+WebSocket API (control plane).

The clean surface the Electron frontend builds against, replacing today's
file polling (gate_health.json / events.db / frames.json). This package is a
THIN, decoupled layer: it does its own bearer-token auth against the account
store and reads the engine's outputs directly, so it never fights the
console's single-session model and adds no load to the detection path.

v0.2 of the contract — this ships the READ-ONLY surface (auth, health,
cameras, events, triage) plus the live WebSocket and a mock server. Write and
config endpoints (rules, routing, schedules, camera CRUD) land once the
contract is frozen with the frontend.

Run:  python -m cvti.api --db runs/site/events.db --site configs/site_live.json
Mock: python -m cvti.api --mock          # canned data, no engine needed
"""

from cvti.api.app import create_app

__all__ = ["create_app"]
