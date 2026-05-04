from __future__ import annotations

import unittest
from typing import List, Tuple

from Driving.drive_to import SafetyConfig, SafetySupervisor


class _Clock:
    def __init__(self, t0: float = 0.0):
        self.t = t0
    def __call__(self) -> float:
        return self.t
    def advance(self, dt: float) -> None:
        self.t += dt


def _ok(x: float, y: float):
    return {"x": x, "y": y, "theta": 0.0, "tracking_ok": True, "tracking": "OK"}

def _lost():
    return {"x": 0.0, "y": 0.0, "theta": 0.0, "tracking_ok": False, "tracking": "LOST"}


class TestSupervisorOKPath(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.logs: List[str] = []
        self.sup = SafetySupervisor(
            cfg=SafetyConfig(),
            now=self.clock,
            log=self.logs.append,
        )

    def test_first_ok_returns_ok(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.assertEqual(self.logs, [])

    def test_consecutive_ok_within_velocity_returns_ok(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(0.067)             # 15 Hz period
        self.assertEqual(self.sup.check(_ok(0.01, 0.0)), "OK")
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(0.02, 0.0)), "OK")
        self.assertEqual(self.logs, [])


class TestSupervisorLostEscalation(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.logs: List[str] = []
        self.sup = SafetySupervisor(
            cfg=SafetyConfig(lost_quiet_sec=0.5, lost_warn_sec=3.0,
                             warn_log_period=0.5),
            now=self.clock,
            log=self.logs.append,
        )

    def test_short_lost_under_quiet_threshold_holds_silently(self):
        self.sup.check(_ok(0.0, 0.0))
        self.clock.advance(0.1)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.clock.advance(0.3)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(self.logs, [])  # silent

    def test_lost_in_warn_window_logs_and_holds(self):
        self.sup.check(_ok(0.0, 0.0))
        self.clock.advance(0.6)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(len(self.logs), 1)
        self.assertIn("tracking lost", self.logs[0])
        # next check 0.1s later: still under warn_log_period → no new log
        self.clock.advance(0.1)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(len(self.logs), 1)
        # now 0.5s after first warn → new log line
        self.clock.advance(0.5)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.assertEqual(len(self.logs), 2)

    def test_lost_beyond_total_threshold_aborts(self):
        self.sup.check(_ok(0.0, 0.0))
        # quiet (0.5) + warn (3.0) = 3.5s total before abort
        self.clock.advance(3.6)
        self.assertEqual(self.sup.check(_lost()), "ABORT")
        self.assertIn("tracking lost", self.sup.reason)

    def test_recovery_clears_lost_state(self):
        self.sup.check(_ok(0.0, 0.0))
        self.clock.advance(1.0)
        self.assertEqual(self.sup.check(_lost()), "HOLD")
        self.clock.advance(0.1)
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.logs.clear()
        # should not log "lost" anymore
        self.clock.advance(0.6)
        # treat next pose as fresh; no new lost
        self.assertEqual(self.sup.check(_ok(0.001, 0.0)), "OK")
        self.assertEqual(self.logs, [])


class TestSupervisorPoseJump(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.logs: List[str] = []
        # max_linear_vel=0.3, jump_factor=3 → at dt=0.067s, threshold = 0.06m
        self.sup = SafetySupervisor(
            cfg=SafetyConfig(max_linear_vel=0.3, jump_factor=3.0,
                             jump_outlier_max=3),
            now=self.clock,
            log=self.logs.append,
        )

    def test_single_jump_holds_then_recovers(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(0.067)                    # threshold ≈ 0.06 m
        self.assertEqual(self.sup.check(_ok(1.0, 0.0)), "HOLD")  # 1m jump
        self.clock.advance(0.067)
        # next plausible pose (close to last_ok=(0,0)) → OK, counter resets
        self.assertEqual(self.sup.check(_ok(0.01, 0.0)), "OK")

    def test_three_consecutive_jumps_abort(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        for i in range(3):
            self.clock.advance(0.067)
            res = self.sup.check(_ok(10.0 + i, 0.0))
            if i < 2:
                self.assertEqual(res, "HOLD")
            else:
                self.assertEqual(res, "ABORT")
        self.assertIn("pose jump", self.sup.reason)

    def test_non_consecutive_jump_does_not_accumulate(self):
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(5.0, 0.0)), "HOLD")     # jump 1
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(0.01, 0.0)), "OK")       # reset
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(5.0, 0.0)), "HOLD")     # jump 1 again, not 2
        self.clock.advance(0.067)
        self.assertEqual(self.sup.check(_ok(0.02, 0.0)), "OK")

    def test_long_dt_disables_jump_check(self):
        # If dt >= 1s (e.g. after a long pause), don't classify as a jump.
        self.assertEqual(self.sup.check(_ok(0.0, 0.0)), "OK")
        self.clock.advance(1.5)
        self.assertEqual(self.sup.check(_ok(2.0, 0.0)), "OK")
