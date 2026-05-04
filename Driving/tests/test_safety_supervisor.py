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
