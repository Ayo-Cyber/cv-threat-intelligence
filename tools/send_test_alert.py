#!/usr/bin/env python3
"""Fire a single test alert through a notifier — verify WhatsApp/Telegram/webhook
end-to-end without waiting for a real detection.

    # WhatsApp (needs Twilio env vars set — see below)
    export TWILIO_ACCOUNT_SID=AC... TWILIO_AUTH_TOKEN=... WHATSAPP_TO=+234...
    python tools/send_test_alert.py --notify whatsapp

    python tools/send_test_alert.py --notify telegram:<token>:<chat_id>
    python tools/send_test_alert.py --notify webhook:https://...
"""
from __future__ import annotations

import argparse
import time

from cvti.serving.alert_sink import build_notifier


def main() -> None:
    p = argparse.ArgumentParser(description="Send one test alert through a notifier.")
    p.add_argument("--notify", default="console",
                   help="console | whatsapp | telegram:<token>:<chat_id> | webhook:<url>")
    p.add_argument("--camera", default="test_cam")
    p.add_argument("--rule", default="shoplifting")
    p.add_argument("--priority", default="high")
    args = p.parse_args()

    notifier = build_notifier(args.notify)
    print(f"[test] notifier = {type(notifier).__name__}")
    event = {
        "ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "camera_id": args.camera, "rule": args.rule, "priority": args.priority,
        "confidence": 0.91, "reason": "CVTI test alert — if you got this, notifications work.",
        "track_id": 1, "zone": None, "object_label": None, "evidence_dir": "(test)",
    }
    notifier.notify(event)
    print("[test] sent. Check the destination (phone / channel).")


if __name__ == "__main__":
    main()
