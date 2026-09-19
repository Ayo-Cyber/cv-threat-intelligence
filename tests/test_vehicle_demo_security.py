"""Security invariants for the committed vehicle demo configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vehicle_demo.json"
TELEGRAM_TOKEN = re.compile(r"\b[0-9]{6,12}:[A-Za-z0-9_-]{30,}\b")


def test_vehicle_demo_config_is_valid_json():
    parsed = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), "vehicle demo config must be a JSON object"


def test_vehicle_demo_notifications_are_local_only_and_credential_free():
    raw = CONFIG.read_text(encoding="utf-8")
    parsed = json.loads(raw)

    if parsed.get("notify") != "console":
        raise AssertionError("vehicle demo notifications must remain local-only")
    if TELEGRAM_TOKEN.search(raw) is not None:
        raise AssertionError("vehicle demo config contains credential-shaped notification data")
