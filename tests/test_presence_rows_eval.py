"""KPI rows 5-7 become runnable: the eval grows a presence lane.

The rows' detector is 'presence' — zone geometry, not a model — and the
harness never built a zone monitor, so person/intrusion/loitering had never
actually been scoreable. These tests hold the new lane: a full-frame zone
stands in for customer-drawn geometry, the loitering row only counts states
past the dwell threshold, the eval rules config actually fires on presence
events, and clips cut from annotated corpora (data/eval_clips/<source>/) are
collected with their source named.
"""
import unittest
from pathlib import Path

from cvti.eval.harness import EvalHarness
from cvti.eval.kpi import presence_harness_kwargs

ROOT = Path(__file__).resolve().parents[1]
CFG = "configs/eval/presence_rows_v1.json"


def _harness(**kw):
    return EvalHarness(detectors=("presence",), config=CFG, **kw)


class PresenceMonitorTest(unittest.TestCase):
    def test_presence_rows_get_a_full_frame_zone(self):
        m = _harness()._presence_monitor()
        from cvti.retail.zones import RetailZoneMonitor
        self.assertIsInstance(m, RetailZoneMonitor)
        self.assertEqual([z.name for z in m.zones], ["frame"])
        self.assertIsNone(m.zones[0].dwell_alert_seconds)

    def test_loitering_gets_the_dwell_filter(self):
        m = _harness(presence_dwell_s=20.0)._presence_monitor()
        self.assertEqual(type(m).__name__, "_LoiterOnly")

        class _S:
            def __init__(self, loitering):
                self.loitering = loitering

        inner = m._inner

        class _FakeInner:
            def update(self, det, ts):
                return [_S(False), _S(True), _S(False)]

        m._inner = _FakeInner()
        out = m.update(None, 0.0)
        self.assertEqual([s.loitering for s in out], [True],
                         "a passer-by below the dwell threshold must not emit")
        self.assertIsNotNone(inner)

    def test_presence_is_not_passed_as_a_state_flag(self):
        # PerCameraState has no `presence` kwarg — passing it through would
        # TypeError on the first clip of the first real run.
        import inspect
        from cvti.serving.camera import PerCameraState
        self.assertNotIn("presence",
                         inspect.signature(PerCameraState.__init__).parameters)


class EngineFiresOnPresenceTest(unittest.TestCase):
    def test_the_eval_config_turns_a_presence_event_into_a_candidate(self):
        from cvti.contracts import RawEvent
        from cvti.rules.customization import CustomizationEngine
        engine = CustomizationEngine(CFG, baseline_path=None)
        ev = RawEvent(detector="presence", active=True,
                      title="PERSON IN ZONE FRAME", level="low", person_id=3,
                      timestamp=1.0,
                      extra={"zone": "frame", "dwell_seconds": 21.0,
                             "loitering": True})
        alerts = engine.evaluate([ev], scene_context={
            "environment_type": "public_space", "scene_description": "test"})
        self.assertEqual([a.rule_name for a in alerts], ["person_present"])
        self.assertEqual(alerts[0].detector, "presence",
                         "the bypass tier keys on the DETECTOR name")


class RowKwargsTest(unittest.TestCase):
    def test_presence_rows_get_the_eval_config(self):
        for key in ("person", "intrusion"):
            kw = presence_harness_kwargs(key, ("presence",))
            self.assertEqual(kw, {"config": CFG}, key)

    def test_loitering_also_gets_the_dwell_threshold(self):
        kw = presence_harness_kwargs("loitering", ("presence",))
        self.assertEqual(kw["presence_dwell_s"], 20.0)

    def test_non_presence_rows_are_untouched(self):
        self.assertEqual(presence_harness_kwargs("theft",
                                                 ("concealment", "video_action")),
                         {})


class CollectorTest(unittest.TestCase):
    def test_cut_corpus_clips_are_collected_with_their_source(self):
        d = ROOT / "data" / "eval_clips" / "_evaltest_"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "loitering_probe_00.mp4"
        f.write_bytes(b"\x00" * 64)
        try:
            from cvti.eval.kpi import collect_clips
            hits = [c for c in collect_clips() if c.path == str(f)]
            self.assertEqual(len(hits), 1)
            self.assertTrue(hits[0].is_threat)
            self.assertEqual(hits[0].kind, "loitering")
            self.assertEqual(hits[0].source, "_evaltest_")
        finally:
            f.unlink()
            d.rmdir()


if __name__ == "__main__":
    unittest.main()
