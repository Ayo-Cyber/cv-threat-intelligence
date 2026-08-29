"""One timed-out verify is not an outage.

Pilot's laptop, 29 Aug, mid-Meet-screen-share over a 90s-median verify: a
single timeout flipped the footer to 'TrueSight UNAVAILABLE — English rules
are paused' and held it there until the next success landed minutes later.
The AI was busy, not dead. Reachability now demands sustained evidence,
scaled to the machine's own measured latency — while a server that has never
answered at all still reads unreachable immediately.
"""
from __future__ import annotations

import unittest

from cvti.serving.pipeline import gate_reachable

NOW = 1_000_000.0


class GateReachabilityTest(unittest.TestCase):

    def test_no_traffic_is_unknown_not_fine(self):
        self.assertIsNone(gate_reachable({}, now=NOW))

    def test_a_recent_success_is_reachable(self):
        s = {"verified": 5, "last_success_at": NOW - 10, "last_unverified_at": NOW - 60}
        self.assertTrue(gate_reachable(s, now=NOW))

    def test_one_fresh_timeout_after_a_success_stays_reachable(self):
        """The pilot's exact frame: success 2 minutes ago, one timeout now."""
        s = {"verified": 4, "unverified": 1, "median_latency_s": 90.2,
             "last_success_at": NOW - 120, "last_unverified_at": NOW - 5}
        self.assertTrue(gate_reachable(s, now=NOW),
                        "one busy-machine timeout declared the AI dead")

    def test_sustained_failure_is_unreachable(self):
        """No success for far longer than the machine's own patience window."""
        s = {"verified": 4, "unverified": 9, "median_latency_s": 90.2,
             "last_success_at": NOW - 900, "last_unverified_at": NOW - 5}
        self.assertFalse(gate_reachable(s, now=NOW))

    def test_a_server_that_never_answered_reads_unreachable_fast(self):
        s = {"errors": 3, "unverified": 3, "last_success_at": 0,
             "last_unverified_at": NOW - 2}
        self.assertFalse(gate_reachable(s, now=NOW))

    def test_patience_scales_with_measured_latency(self):
        slow = {"verified": 1, "unverified": 1, "median_latency_s": 90.0,
                "last_success_at": NOW - 300, "last_unverified_at": NOW - 5}
        fast = dict(slow, median_latency_s=10.0)
        self.assertTrue(gate_reachable(slow, now=NOW),
                        "a 90s-median machine deserves a 360s window")
        self.assertFalse(gate_reachable(fast, now=NOW),
                         "a 10s-median machine 300s without success IS down")
