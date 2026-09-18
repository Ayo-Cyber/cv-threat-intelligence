"""The stdio transport exposes object-watch operations without changing semantics."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from Frontend.bridge import METHODS, dispatch
from cvti.app.console_backend import ConsoleBackend
from cvti.security.permissions import PermissionDenied


OBJECT_WATCH_CALLS = {
    "object_targets": (),
    "create_object_target": (
        "forklift-7", "Forklift 7", "vehicle", ["lift"], 0.81, ["yard"],
        "yellow forklift",
    ),
    "set_object_watch_runtime_config": ({"backend": "siglip", "device": "cpu"},),
    "reembed_object_targets": (),
    "add_object_example": (
        "forklift-7", "data:image/png;base64,cG5n", [1, 2, 30, 40], "upload",
        "pixel_xyxy", False,
    ),
    "review_object_example": ("forklift-7", "example-1", True),
    "object_example_preview": ("forklift-7", "example-1"),
    "activate_object_target": ("forklift-7",),
    "deactivate_object_target": ("forklift-7",),
    "object_watch_job_status": ("job-1",),
    "set_object_watch_rule": ("camera-1", "forklift-7", True, "yard"),
}


class RecordingBackend:
    current_user: SimpleNamespace | None = SimpleNamespace(role="owner")

    def __init__(self):
        self.calls = []

    def _record(self, method, *args):
        self.calls.append((method, args))
        return {"method": method, "args": args}

    def object_targets(self):
        return self._record("object_targets")

    def create_object_target(self, object_id, label, category, aliases=None,
                             min_similarity=0.72, allowed_zone_ids=None,
                             grounding_description=""):
        return self._record(
            "create_object_target", object_id, label, category, aliases,
            min_similarity, allowed_zone_ids, grounding_description,
        )

    def set_object_watch_runtime_config(self, config):
        return self._record("set_object_watch_runtime_config", config)

    def reembed_object_targets(self, model=None):
        return self._record("reembed_object_targets", model)

    def add_object_example(self, object_id, image_b64, bbox, source,
                           bbox_format="legacy", negative=False):
        return self._record(
            "add_object_example", object_id, image_b64, bbox, source,
            bbox_format, negative,
        )

    def review_object_example(self, object_id, example_id, reviewed=True):
        return self._record(
            "review_object_example", object_id, example_id, reviewed,
        )

    def object_example_preview(self, object_id, example_id):
        return self._record("object_example_preview", object_id, example_id)

    def activate_object_target(self, object_id):
        return self._record("activate_object_target", object_id)

    def deactivate_object_target(self, object_id):
        return self._record("deactivate_object_target", object_id)

    def object_watch_job_status(self, job_id):
        return self._record("object_watch_job_status", job_id)

    def set_object_watch_rule(self, camera_id, object_id, enabled, zone_id=None):
        return self._record(
            "set_object_watch_rule", camera_id, object_id, enabled, zone_id,
        )


class OperatorBackend:
    current_user = SimpleNamespace(role="operator")
    _role = ConsoleBackend._role
    _require = ConsoleBackend._require
    create_object_target = ConsoleBackend.create_object_target
    set_object_watch_runtime_config = ConsoleBackend.set_object_watch_runtime_config
    reembed_object_targets = ConsoleBackend.reembed_object_targets
    add_object_example = ConsoleBackend.add_object_example
    review_object_example = ConsoleBackend.review_object_example
    activate_object_target = ConsoleBackend.activate_object_target
    deactivate_object_target = ConsoleBackend.deactivate_object_target
    object_watch_job_status = ConsoleBackend.object_watch_job_status
    set_object_watch_rule = ConsoleBackend.set_object_watch_rule


class FrontendBridgeObjectWatchTests(unittest.TestCase):
    def test_dispatch_preserves_frontend_positional_arguments(self):
        backend = RecordingBackend()

        for method, args in OBJECT_WATCH_CALLS.items():
            self.assertIn(method, METHODS)
            result = dispatch(backend, method, list(args))
            expected = args if method != "reembed_object_targets" else (None,)
            self.assertEqual(result, {"method": method, "args": expected})

        self.assertEqual(
            [method for method, _ in backend.calls], list(OBJECT_WATCH_CALLS),
        )

    def test_dispatch_requires_a_signed_in_user(self):
        backend = RecordingBackend()
        backend.current_user = None

        for method, args in OBJECT_WATCH_CALLS.items():
            with self.assertRaisesRegex(PermissionError, "Sign in"):
                dispatch(backend, method, list(args))

        self.assertEqual(backend.calls, [])

    def test_mutations_keep_console_backend_role_enforcement(self):
        backend = OperatorBackend()
        mutations = set(OBJECT_WATCH_CALLS) - {
            "object_targets", "object_example_preview",
        }

        for method in mutations:
            with self.assertRaisesRegex(PermissionDenied, "role 'operator' may not"):
                dispatch(backend, method, list(OBJECT_WATCH_CALLS[method]))
