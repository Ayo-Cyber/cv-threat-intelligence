"""Argus Engine HTTP+WebSocket API (control plane).

The clean surface the Electron frontend builds against, replacing today's
file polling (gate_health.json / events.db / frames.json). This package is a
thin control-plane layer: it does its own bearer-token auth against the account
store, reads canonical engine outputs directly, and delegates configuration and
control operations through the existing permission-checked backend.

v0.2 ships auth, health, cameras, events, triage, hierarchy, configuration and
engine-control endpoints, plus the live WebSocket and a mock server.

Run:  python -m cvti.api --db runs/site/events.db --site configs/site_live.json
Mock: python -m cvti.api --mock          # canned data, no engine needed
"""

from cvti.api.app import create_app

__all__ = ["create_app"]
