from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvti.contracts import RawEvent
from cvti.rules.customization import CustomizationEngine, _logic_satisfied, PRIORITY_ORDER

HIGH = PRIORITY_ORDER["high"]
MED = PRIORITY_ORDER["medium"]


def _ev(detector, level="medium", title="X", signal_type=None):
    extra = {"signal_type": signal_type} if signal_type else {}
    return RawEvent(detector=detector, active=True, title=title, level=level,
                    timestamp=1.0, extra=extra)


class LogicOpTests(unittest.TestCase):
    def test_one_high_or_two_medium(self):
        f = _logic_satisfied
        self.assertTrue(f("one_high_or_two_medium", [HIGH], 6))          # one high
        self.assertTrue(f("one_high_or_two_medium", [MED, MED], 6))      # two medium
        self.assertFalse(f("one_high_or_two_medium", [MED], 6))          # one medium only
    def test_at_least_and_all(self):
        self.assertTrue(_logic_satisfied("at_least_2", [MED, MED], 3))
        self.assertFalse(_logic_satisfied("at_least_2", [MED], 3))
        self.assertTrue(_logic_satisfied("all", [MED, MED, MED], 3))
        self.assertFalse(_logic_satisfied("all", [MED, MED], 3))


class CompoundRecipeTests(unittest.TestCase):
    def setUp(self):
        self.engine = CustomizationEngine("configs/compound_recipes_v1.json")

    def test_armed_robbery_fires_on_one_high_signal(self):
        alerts = self.engine.evaluate([_ev("weapons", level="critical", title="GUN")])
        ar = [a for a in alerts if a.rule_name == "armed_robbery"]
        self.assertEqual(len(ar), 1)
        self.assertEqual(ar[0].detector, "compound")
        self.assertEqual(ar[0].priority, "critical")
        self.assertIsNotNone(ar[0].question)          # recipe gate_question threaded through

    def test_violent_theft_needs_two_signals(self):
        one = self.engine.evaluate([_ev("concealment", level="high")])
        self.assertFalse(any(a.rule_name == "violent_theft" for a in one))
        two = self.engine.evaluate([_ev("concealment", level="high"), _ev("running", level="medium")])
        self.assertTrue(any(a.rule_name == "violent_theft" for a in two))

    def test_video_action_signal_type_matches(self):
        # a weak VideoMAE "violence_candidate" should count as the violence signal
        ev = _ev("video_action", level="medium", signal_type="violence_candidate")
        alerts = self.engine.evaluate([ev, _ev("running", level="medium")])
        # violence(1) + running(1) -> two medium -> armed_robbery fires
        self.assertTrue(any(a.rule_name == "armed_robbery" for a in alerts))


if __name__ == "__main__":
    unittest.main()


class SimpleRuleQuestionTest(unittest.TestCase):
    """A rule's plain-English gate_question must reach the VLM (demo-day bug).

    Compound recipes always carried it; simple trigger rules silently dropped
    it — the customer's sentence never reached the model, and the gate fell
    back to a generic question while the UI implied otherwise.
    """

    def test_gate_question_rides_the_candidate(self):
        import json
        import tempfile
        from cvti.contracts import RawEvent
        from cvti.rules.customization import CustomizationEngine
        rule = {"use_case_id": "t", "rules": [{
            "name": "stated", "trigger": {"detector": "presence"},
            "priority": "high",
            "gate_question": "Is someone waving both arms above their head?"}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(rule, f)
        eng = CustomizationEngine(f.name)
        got = eng.evaluate([RawEvent(detector="presence", active=True,
                                     title="p", level="low")], scene_context={})
        self.assertEqual(got[0].question,
                         "Is someone waving both arms above their head?")

    def test_rules_without_a_question_keep_the_default(self):
        import json
        import tempfile
        from cvti.contracts import RawEvent
        from cvti.rules.customization import CustomizationEngine
        rule = {"use_case_id": "t", "rules": [{
            "name": "plain", "trigger": {"detector": "presence"}}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(rule, f)
        eng = CustomizationEngine(f.name)
        got = eng.evaluate([RawEvent(detector="presence", active=True,
                                     title="p", level="low")], scene_context={})
        self.assertIsNone(got[0].question)


class MultipleEnglishRulesTest(unittest.TestCase):
    """User feedback 23 Aug: 'it seems to only be tailored to hoodie' — one
    sentence per camera overwrote the last. Sentences now accumulate, each its
    own rule with its own question, and changes hot-apply to a running engine."""

    def _backend(self, tmp):
        import json, os, sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from _backend_helper import signed_in
        (Path(tmp) / "site.json").write_text(json.dumps(
            {"cameras": [{"id": "c1", "source": "0"}]}))
        os.chdir(tmp)
        return signed_in("owner", site_path=str(Path(tmp) / "site.json"),
                         db_path=str(Path(tmp) / "events.db"), enable_demo=False)

    def test_sentences_accumulate_instead_of_overwriting(self):
        import json, os, tempfile
        from pathlib import Path
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                be = self._backend(tmp)
                be.add_custom_rule("c1", "Is anyone wearing a black hoodie?")
                be.add_custom_rule("c1", "Is anyone carrying a ladder?")
                cam = json.loads((Path(tmp) / "site.json").read_text())["cameras"][0]
                rules = json.loads(Path(cam["config"]).read_text())["rules"]
                qs = [r["gate_question"] for r in rules if r.get("gate_question")]
                self.assertEqual(len(qs), 2, "the second sentence overwrote the first")
                self.assertIn("hoodie", qs[0]); self.assertIn("ladder", qs[1])
            finally:
                os.chdir(cwd)

    def test_removing_one_sentence_keeps_the_others(self):
        import json, os, tempfile
        from pathlib import Path
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                be = self._backend(tmp)
                be.add_custom_rule("c1", "Is anyone wearing a black hoodie?")
                be.add_custom_rule("c1", "Is anyone carrying a ladder?")
                be.remove_custom_rule("c1", "Is anyone wearing a black hoodie?")
                cam = json.loads((Path(tmp) / "site.json").read_text())["cameras"][0]
                rules = json.loads(Path(cam["config"]).read_text())["rules"]
                qs = [r["gate_question"] for r in rules if r.get("gate_question")]
                self.assertEqual(len(qs), 1)
                self.assertIn("ladder", qs[0])
            finally:
                os.chdir(cwd)

    def test_legacy_single_rule_is_migrated_not_lost(self):
        import json, os, tempfile
        from pathlib import Path
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                be = self._backend(tmp)
                # a site file from before the list existed
                site = json.loads((Path(tmp) / "site.json").read_text())
                site["cameras"][0]["custom_rule"] = {"question": "Old sentence?", "dwell": 4}
                (Path(tmp) / "site.json").write_text(json.dumps(site))
                be.add_custom_rule("c1", "New sentence?")
                cam = json.loads((Path(tmp) / "site.json").read_text())["cameras"][0]
                self.assertNotIn("custom_rule", cam, "legacy field left to shadow the list")
                qs = [r["question"] for r in cam["custom_rules"]]
                self.assertEqual(qs, ["Old sentence?", "New sentence?"])
            finally:
                os.chdir(cwd)


class HotReloadTest(unittest.TestCase):
    """'It should kick off automatically, not until a Start monitoring' —
    rules and zones are JSON, not models: the running engine swaps them in."""

    def test_refresh_swaps_the_rules_engine_in_place(self):
        import json, tempfile
        from pathlib import Path
        from cvti.serving.camera import build_camera_states, refresh_camera_rules
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "rules.json"
            cfg.write_text(json.dumps({"use_case_id": "t", "rules": [{
                "name": "r1", "trigger": {"detector": "presence"},
                "gate_question": "Old question?"}]}))
            site = {"cameras": [{"id": "c1", "source": "x.mp4", "config": str(cfg)}]}
            states = build_camera_states(site)
            st = states["c1"]["state"]
            ev = RawEvent(detector="presence", active=True, title="p", level="low")
            self.assertEqual(st.engine.evaluate([ev], scene_context={})[0].question,
                             "Old question?")
            # the operator types a new sentence -> the file changes on disk
            cfg.write_text(json.dumps({"use_case_id": "t", "rules": [{
                "name": "r1", "trigger": {"detector": "presence"},
                "gate_question": "New question?"}]}))
            refresh_camera_rules(st, site["cameras"][0])
            self.assertEqual(st.engine.evaluate([ev], scene_context={})[0].question,
                             "New question?", "the running engine kept the old rules")

    def test_the_engine_watcher_wires_the_refresh(self):
        src = (Path(__file__).resolve().parents[1] / "cvti/serving/pipeline.py").read_text()
        self.assertIn("_rule_fingerprint", src)
        self.assertIn("refresh_camera_rules(states[cid], cam, baseline_config)", src)
