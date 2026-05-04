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
