from __future__ import annotations

import unittest

from Driving.wheel_motor import WheelMotorClient, WheelMotorConfig


class TestWheelMotorConfig(unittest.TestCase):
    def test_defaults(self):
        c = WheelMotorConfig()
        self.assertEqual(c.port, "/dev/ttyACM0")
        self.assertEqual(c.baud, 115200)
        self.assertEqual(c.max_wheel_mrad_s, 30000)
        self.assertEqual(c.deadzone_mrad_s, 5)
        self.assertEqual(c.direction_signs, (+1, +1))
        self.assertFalse(c.verbose)
        self.assertFalse(c.dry_run)


class TestWheelMotorClientConstruction(unittest.TestCase):
    def test_can_instantiate_with_dry_run(self):
        client = WheelMotorClient(WheelMotorConfig(dry_run=True))
        self.assertTrue(client.cfg.dry_run)
        self.assertEqual(client.sent_lines, [])
