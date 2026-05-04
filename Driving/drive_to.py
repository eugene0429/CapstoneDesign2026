"""
Phase-1 Driving Pipeline — standalone runner.

Reads pose from ORB-SLAM3 in real time, runs the existing DrivingController
to compute (ω_L, ω_R), and streams them to OpenRB-150 over the wheel-motor
serial protocol. Stops cleanly on goal-reach, timeout, SLAM failure, or Ctrl-C.

Spec: docs/superpowers/specs/2026-05-04-driving-pipeline-design.md

Usage
-----
    python Driving/drive_to.py --x 3 --y 2
    python Driving/drive_to.py --x 3 --y 2 --dry-run
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


# ──────────────────────────── safety ────────────────────────────
@dataclass
class SafetyConfig:
    lost_quiet_sec:    float = 0.5
    lost_warn_sec:     float = 3.0     # logging window above quiet
    # ABORT after lost_quiet_sec + lost_warn_sec total

    jump_factor:       float = 3.0
    jump_outlier_max:  int   = 3
    max_linear_vel:    float = 0.3     # m/s — matches ControllerConfig.max_speed

    warn_log_period:   float = 0.5     # min interval between warn logs


class SafetySupervisor:
    """Per-frame safety check. Returns "OK" | "HOLD" | "ABORT".

    Test seam: `now` defaults to time.monotonic; `log` defaults to print.
    Tests inject deterministic versions.
    """

    def __init__(
        self,
        cfg: Optional[SafetyConfig] = None,
        now: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = print,
    ):
        self.cfg = cfg if cfg is not None else SafetyConfig()
        self._now = now
        self._log = log
        self.reason: str = ""

        # state
        self._last_ok: Optional[Tuple[float, float, float]] = None  # (x, y, t)
        self._lost_since: Optional[float] = None
        self._consec_outliers: int = 0
        self._last_warn_at: float = -1e9

    def check(self, pose: Optional[Dict]) -> str:
        c = self.cfg
        t = self._now()

        # Branch A: tracking lost or pose unavailable
        if pose is None or not pose.get("tracking_ok", False):
            if self._lost_since is None:
                # back-date to last known-good timestamp so the quiet/warn
                # window counts from the moment tracking was last confirmed
                last_t = self._last_ok[2] if self._last_ok is not None else t
                self._lost_since = last_t
            dur = t - self._lost_since
            if dur < c.lost_quiet_sec:
                return "HOLD"
            if dur < c.lost_quiet_sec + c.lost_warn_sec:
                if t - self._last_warn_at >= c.warn_log_period:
                    self._log(f"[WARN] tracking lost {dur:.1f}s")
                    self._last_warn_at = t
                return "HOLD"
            self.reason = (
                f"tracking lost {dur:.1f}s "
                f"(>= {c.lost_quiet_sec + c.lost_warn_sec:.1f}s)"
            )
            return "ABORT"

        # pose-jump rejection (only when we have a prior accepted pose)
        if self._last_ok is not None:
            ox, oy, ot = self._last_ok
            dt = t - ot
            if 0.0 < dt < 1.0:
                jump = math.hypot(float(pose["x"]) - ox, float(pose["y"]) - oy)
                threshold = c.max_linear_vel * dt * c.jump_factor
                if jump > threshold:
                    self._consec_outliers += 1
                    if self._consec_outliers >= c.jump_outlier_max:
                        self.reason = (
                            f"pose jump x{c.jump_outlier_max} "
                            f"(last={jump:.2f}m in {dt*1000:.0f}ms)"
                        )
                        return "ABORT"
                    return "HOLD"

        # accepted
        self._last_ok = (float(pose["x"]), float(pose["y"]), t)
        self._lost_since = None
        self._consec_outliers = 0
        return "OK"


# ──────────────────────────── runner ────────────────────────────
@dataclass
class RunArgs:
    x: float
    y: float
    rate: float = 15.0
    timeout: float = 60.0
    port: str = "/dev/ttyACM0"
    baud: int = 115200
    dry_run: bool = False
    verbose: bool = False


def _log_status(t_elapsed, pose, out, log: Callable[[str], None]) -> None:
    log(
        f"  [{t_elapsed:5.2f}s]  pose=("
        f"{pose['x']:+.2f}, {pose['y']:+.2f}, "
        f"{math.degrees(pose['theta']):+6.1f}°)  "
        f"dist={out['distance']:.2f}  v={out['v']:.2f}  "
        f"ω_L/R=({out['wheel_omega_left']:+.2f}, "
        f"{out['wheel_omega_right']:+.2f})"
    )


def _run_loop(
    args: RunArgs,
    localizer,
    controller,
    motor,
    supervisor: SafetySupervisor,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> int:
    """Main 15 Hz control loop. Returns a process exit code:
    0 = reached, 1 = timeout, 2 = supervisor ABORT.
    """
    period = 1.0 / args.rate
    t_start = now()
    deadline = t_start + args.timeout
    last_log_at = -1e9

    while now() < deadline:
        t0 = now()
        pose = localizer.get_pose()
        action = supervisor.check(pose)

        if action == "ABORT":
            log(f"[ABORT] {supervisor.reason}")
            motor.drive(0.0, 0.0)
            return 2

        if action == "HOLD":
            motor.drive(0.0, 0.0)
        else:  # OK
            out = controller.compute(
                pose["x"], pose["y"], pose["theta"], args.x, args.y)
            if out["reached"]:
                motor.drive(0.0, 0.0)
                log(f"reached @ dist={out['distance']:.3f}m")
                return 0
            motor.drive(out["wheel_omega_left"], out["wheel_omega_right"])
            if t0 - last_log_at >= 0.5:
                _log_status(t0 - t_start, pose, out, log)
                last_log_at = t0

        elapsed = now() - t0
        sleep(max(0.0, period - elapsed))

    log(f"[TIMEOUT] {args.timeout:.1f}s elapsed, goal not reached")
    motor.drive(0.0, 0.0)
    return 1
