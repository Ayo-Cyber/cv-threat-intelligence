"""The eval harness's loitering wrapper must accept whatever the engine passes.

PerCameraState.process calls `zone_monitor.update(tracked, ts, frame_hw=...)`
and has done since #145 (doorway polygons). The harness wraps the monitor to
keep only dwell-crossing states, and that wrapper used to declare
`update(self, detections, timestamp)` — so every loitering evaluation clip
died on `TypeError: update() got an unexpected keyword argument 'frame_hw'`
-- and once that was forwarded, on `AttributeError: drain_exits`, the two-tier
exit drain the same PR added.

Nothing failed loudly: the KPI row was already below its n floor, so its
verdict read SMOKE either way, and clip errors do not fail a scorecard run.
The loitering row was silently unmeasurable for four releases.
"""
from __future__ import annotations

import inspect
import unittest

from cvti.eval.harness import EvalHarness


class TheLoiteringWrapperForwardsWhatTheEngineSends(unittest.TestCase):
    def _wrapper(self):
        harness = EvalHarness(presence_dwell_s=5.0)
        return harness._presence_monitor()

    def test_it_accepts_frame_hw(self):
        wrapper = self._wrapper()
        self.assertNotIsInstance(
            wrapper, type(None), "a dwell harness must wrap the monitor")
        signature = inspect.signature(wrapper.update)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        self.assertTrue(
            accepts_kwargs or "frame_hw" in signature.parameters,
            f"the wrapper's update{signature} cannot take frame_hw=, which "
            "PerCameraState.process passes on every frame",
        )

    def test_it_passes_frame_hw_through_and_still_filters(self):
        class Inner:
            def __init__(self):
                self.seen = []

            def update(self, detections, timestamp, **kwargs):
                self.seen.append(kwargs)
                passer_by = type("S", (), {"loitering": False})()
                loiterer = type("S", (), {"loitering": True})()
                return [passer_by, loiterer]

        wrapper = self._wrapper()
        inner = Inner()
        wrapper._inner = inner
        kept = wrapper.update(None, 1.0, frame_hw=(480, 640))
        self.assertEqual(inner.seen, [{"frame_hw": (480, 640)}],
                         "frame_hw must reach the real monitor")
        self.assertEqual([state.loitering for state in kept], [True],
                         "only dwell-crossing states survive the wrapper")


    def test_it_proxies_the_rest_of_the_monitor(self):
        """process() also calls drain_exits(); the wrapper must not hide it."""
        wrapper = self._wrapper()
        for name in ("drain_exits", "zones"):
            self.assertTrue(
                hasattr(wrapper, name),
                f"PerCameraState.process reaches for .{name} on the zone "
                "monitor; the loitering wrapper hides it",
            )


if __name__ == "__main__":
    unittest.main()
