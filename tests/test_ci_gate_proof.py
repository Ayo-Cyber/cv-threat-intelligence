"""TEMPORARY — deliberately failing test to prove CI blocks a red suite.

Reverted immediately after the run. EP-00-T1 acceptance.
"""
import unittest


class CiGateProof(unittest.TestCase):
    def test_this_must_fail(self):
        self.assertEqual(1, 2, "deliberate failure — proving the CI gate blocks the build")
