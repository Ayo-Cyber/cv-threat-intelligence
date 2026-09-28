"""Did the KPI run's alerts reach Telegram? Read the engine log and say so.

    python tests/e2e/kpi_assert.py <engine.log> --chats 123,-456

Passes only when EVERY alert kind (person entered / exited, vehicle entered /
exited) has at least one `[notify telegram] delivered ...` line, and EVERY
chat id received at least one delivery. A `[CONFIRMED]` / `[NOTIFY]` line alone is the engine
deciding to send; only `delivered` is the Bot API saying yes -- the
difference cost a day once (see check-checkout-and-delivery memory).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

RULES = ("person_entered", "person_exited", "vehicle_entered", "vehicle_exited")
DELIVERED = re.compile(r"\[notify telegram\] delivered (?P<rule>[a-z_]+) on (?P<camera>\S+) to chat (?P<chat>-?\d+)")
# What the engine wrote when it decided to alert: deterministic detectors
# are auto-confirmed ([CONFIRMED]); critical ones go out provisionally
# ([PROVISIONAL] then [NOTIFY]). Any of them is "raised".
NOTIFY = re.compile(r"\[(?:NOTIFY|CONFIRMED|PROVISIONAL)\] (?P<camera>\S+) :: (?P<rule>[a-z_ ]+?) \(")
QUEUED = re.compile(r"(?P<n>\d+) alert\(s\) still queued")


def verdict(log_text: str, chats: list[str]) -> tuple[bool, str]:
    delivered = Counter()
    per_chat = Counter()
    for m in DELIVERED.finditer(log_text):
        delivered[m.group("rule")] += 1
        per_chat[m.group("chat")] += 1
    raised = Counter(m.group("rule").strip() for m in NOTIFY.finditer(log_text))
    queued = sum(int(m.group("n")) for m in QUEUED.finditer(log_text))
    lines = ["rule              raised  delivered"]
    ok = True
    for r in RULES:
        flag = "" if delivered[r] else "   <-- none delivered"
        ok = ok and delivered[r] > 0
        lines.append(f"{r:<17} {raised[r]:>6} {delivered[r]:>10}{flag}")
    lines.append("")
    for c in chats:
        flag = "" if per_chat[c] else "   <-- nothing reached this chat"
        ok = ok and per_chat[c] > 0
        lines.append(f"chat {c:<14} {per_chat[c]:>4} delivered{flag}")
    if queued:
        lines.append(f"\n{queued} alert(s) were still queued when the engine stopped (rate limit); "
                     "not a failure by itself.")
    lines.append("\nVERDICT: " + ("PASS" if ok else "FAIL"))
    return ok, "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--chats", default="", help="comma-separated chat ids that must each get a delivery")
    a = p.parse_args(argv)
    chats = [c.strip() for c in a.chats.split(",") if c.strip()]
    ok, report = verdict(Path(a.log).read_text(errors="replace"), chats)
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
