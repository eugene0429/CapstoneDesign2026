"""
Wheel Motor Client — Pi5 ↔ OpenRB-150 serial driver for differential-drive wheel
angular velocities.

Wire protocol (full spec in docs/superpowers/specs/2026-05-04-driving-pipeline-design.md §3):

    Pi → OpenRB                   OpenRB → Pi
    ─────────────                 ─────────────
    PING\\n                        PONG\\n           (sync, health check)
    DRIVE <wL> <wR>\\n              (no reply)       (fire-and-forget @ 15 Hz)
    STOP\\n                        OK\\n             (sync, terminal stop)

`<wL>`, `<wR>` are signed integer mrad/s (rad/s × 1000), clamped to ±30000.
The OpenRB firmware is expected to autonomously zero both motors if no DRIVE
packet arrives within 200 ms (watchdog) — this script is correct only against
that firmware contract.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ─────────────────────────── protocol constants ───────────────────────────
_PING = "PING"
_PONG = "PONG"
_STOP = "STOP"
_OK = "OK"
_DRIVE_FMT = "DRIVE {wL} {wR}"


# ──────────────────────────────── config ──────────────────────────────────
@dataclass
class WheelMotorConfig:
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    open_settle_sec: float = 2.0
    sync_read_timeout_sec: float = 1.0
    write_timeout_sec: float = 0.5

    max_wheel_mrad_s: int = 30000
    deadzone_mrad_s: int = 5

    direction_signs: Tuple[int, int] = (+1, +1)

    verbose: bool = False
    dry_run: bool = False


# ──────────────────────────────── client ──────────────────────────────────
class WheelMotorClient:
    """Streaming wheel-velocity client. Use as a context manager."""

    def __init__(self, cfg: Optional[WheelMotorConfig] = None):
        self.cfg = cfg if cfg is not None else WheelMotorConfig()
        self._ser = None
        self.sent_lines: List[str] = []   # populated only when dry_run=True
