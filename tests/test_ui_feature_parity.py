"""The new desktop UI must be able to reach what the old console could.

v1.8.13 replaced the PyQt console with the React shell. The engine kept every
capability, but 42 of them had no route, no client operation and no screen —
model downloads, backups, audit export, account recovery, reports. They were
not removed; they became unreachable, and nothing failed because nothing
checked. This checks.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def old_console_methods() -> set[str]:
    src = (ROOT / "cvti/app/bridge.py").read_text()
    return {_snake(m) for m in re.findall(r"def ([A-Za-z_][A-Za-z_0-9]*)", src)
            if not m.startswith("_")}


def api_routes() -> set[str]:
    src = (ROOT / "cvti/api/writes.py").read_text()
    return set(re.findall(r'R\("([a-z_0-9]+)"', src))


def ipc_allowlist() -> set[str]:
    src = (ROOT / "Frontend/electron/main.ts").read_text()
    block = src[src.index("const methods"):]
    return set(re.findall(r'"([a-z_0-9]+)"', block[:block.index("])")]))


def client_operations() -> set[str]:
    src = (ROOT / "Frontend/electron/api-client.ts").read_text()
    block = src[src.index("const operations"):]
    depth = 0
    for i, ch in enumerate(block):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = block[:i]
                break
    ops = set(re.findall(r"^\s{2}([a-z_0-9]+):", block, re.M))
    # handled explicitly ahead of the table in invoke()
    return ops | {"auth_state", "sign_in", "create_first_owner", "sign_out",
                  "event_clip", "live_start", "live_stop"}


def stdio_methods() -> set[str]:
    src = (ROOT / "Frontend/bridge.py").read_text()
    return set(re.search(r"METHODS = set\('([^']+)'", src, re.S).group(1).split())


class FeatureParity(unittest.TestCase):
    # Genuinely obsolete with the PyQt shell: Qt-only plumbing, or replaced.
    RETIRED = {
        # Qt-only plumbing, never a product capability.
        "accept",
        # Bridge-only helpers that never existed on ConsoleBackend, so there is
        # no capability behind them (verified: hasattr(ConsoleBackend, ...) is
        # False for both).
        "test_url", "list_events_lite",
        # Superseded: per-camera streams start and stop through camera_stream,
        # and the API client rejects these by name with that explanation.
        "live_stop",
        # Superseded by the /reports/* family.
        "review", "handover_pdf",
    }

    def test_every_old_console_capability_is_reachable(self):
        old = old_console_methods() - self.RETIRED
        allow = ipc_allowlist()
        unreachable = sorted(m for m in old if m not in allow)
        # Anything still here is a capability the product HAS and the UI cannot
        # use. Add a route + client operation + allowlist entry, or list it in
        # RETIRED with a reason.
        self.assertEqual(
            unreachable, [],
            "backend capabilities the desktop UI cannot reach: " + ", ".join(unreachable))

    def test_the_allowlist_and_the_transports_agree(self):
        allow, ops, stdio = ipc_allowlist(), client_operations(), stdio_methods()
        self.assertEqual(sorted(allow - ops), [],
                         "allowlisted but the API client cannot perform it — "
                         "invoke() would throw 'Unsupported engine operation'")
        self.assertEqual(sorted(allow - stdio), [],
                         "allowlisted for the API transport but missing from the "
                         "stdio bridge, so it breaks under ARGUS_TRANSPORT=bridge")

    def test_every_route_targets_a_real_backend_method(self):
        from cvti.app.console_backend import ConsoleBackend
        missing = sorted(m for m in api_routes() if not hasattr(ConsoleBackend, m))
        self.assertEqual(missing, [], f"routes with no backend method: {missing}")

    def test_static_routes_are_not_shadowed_by_a_parameterised_sibling(self):
        """`/events/counts` lost to `/events/{event_id}` and 404'd as "no such
        event 'counts'"; `/cameras/discovery` did the same. Same collision that
        hid /cameras/presets (#141). FastAPI matches in declaration order, and
        app.py declares its routes before register_writes runs.

        Verb matters: a POST is never shadowed by a GET, which is why
        POST /cameras/probe reaches its handler (verified live: 400, not 404).
        """
        src = (ROOT / "cvti/api/writes.py").read_text()
        rows = re.findall(r'R\("[a-z_0-9]+",\s*"([A-Z]+)",\s*"([^"]+)"', src)
        app = (ROOT / "cvti/api/app.py").read_text()
        declared = {(v.upper(), p.replace("{{", "{").replace("}}", "}"))
                    for v, p in re.findall(
                        r'@app\.(get|post|put|delete)\(f"\{API_PREFIX\}([^"]+)"', app)}
        # A static row app.py also declares itself is answered by app.py's own
        # handler, registered first — that is how #141 fixed /cameras/presets.
        answered = {p for _, p in declared}
        clashes = []
        for verb, path in rows:
            if "{" in path or path in answered:
                continue
            for dverb, other in declared:
                if dverb != verb or "{" not in other:
                    continue
                a, b = path.strip("/").split("/"), other.strip("/").split("/")
                if len(a) == len(b) and all(
                        x == y or y.startswith("{") for x, y in zip(a, b)):
                    clashes.append(f"{verb} {path} is shadowed by {dverb} {other}")
        self.assertEqual(clashes, [], "; ".join(clashes))

    def test_the_system_panel_exposes_what_has_no_screen(self):
        panel = (ROOT / "Frontend/src/components/SystemPanel.tsx").read_text()
        src = "\n".join(p.read_text() for p in (ROOT / "Frontend/src").rglob("*.ts*"))
        called = set(re.findall(r'["\']([a-z_0-9]+)["\']', src))
        orphans = sorted(m for m in ipc_allowlist() if m not in called)
        self.assertEqual(
            orphans, [],
            "allowlisted but no screen calls it — add it to SystemPanel: "
            + ", ".join(orphans))
        self.assertIn("pull_model", panel,
                      "the first-run model download must be reachable from the UI")


if __name__ == "__main__":
    unittest.main()
