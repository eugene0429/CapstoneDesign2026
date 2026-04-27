"""
Capstone 2026 — Full Pipeline Orchestrator
==========================================

YOLO 학습 전 단계에서 모든 production 모듈의 연동과 Phase 전환을
점검하기 위한 통합 실행기.

연동 대상
--------
- DrivingController       (Driving/controller.py)
- LevelingIK              (LevelingPlatform/leveling_ik.py)
- OrbSlamLocalizer        (perception/vio/orbslam_localizer.py)   [--mode real]
- DummyTargetProvider     (perception/detection/dummy_detector.py)

실행 모드
--------
--mode sim   : 카메라/모터 없이 Pure Python 시뮬레이션 (어디서나 실행 가능)
--mode real  : 실제 RealSense + ORB-SLAM3 측위, 모터는 stub 출력
              (Pi5 + 카메라 장착 환경 필요)

Phase 전환
---------
[Phase 1: Driving]
    DummyTargetProvider.get_phase1_target() → world (x, y)
    → DrivingController.compute(pose, target) → (ω_L, ω_R)
    → 도달 (out["reached"]) 판정 시 Phase 2 로 전이

[Phase 2: Aiming & Strike ×2]
    카메라 90° 틸트 → DummyTargetProvider.get_phase2_target() → plate-frame (x, y, z)
    → LevelingIK.aim_at(target) → 모터 각도 → FIRE
    → 매 타격 직전 target 재추정 (벨 진동 대응)

CLI
---
    python3 pipeline.py                              # 기본 sim 모드
    python3 pipeline.py --phase1-x 4 --phase1-y 3    # 타겟 위치 변경
    python3 pipeline.py --mode real                  # 실제 카메라 + ORB-SLAM3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# ── 패키지 경로 추가 (root 에서 실행되는 통합 스크립트) ──
ROOT = Path(__file__).resolve().parent
for sub in ("Driving", "LevelingPlatform", "perception"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from controller import ControllerConfig, DrivingController          # noqa: E402
from leveling_ik import LevelingConfig, LevelingIK                  # noqa: E402
from detection.dummy_detector import (                              # noqa: E402
    DummyTargetConfig, DummyTargetProvider,
)


# ─────────────────────────────────────────────────────────────────────
# Robot adapters — pose 입력원 + 모터 출력 stub 을 한 인터페이스로 묶음
# ─────────────────────────────────────────────────────────────────────
class SimulatedRobot:
    """순수 Python 시뮬레이션. 차동 구동 정기구학 + 무손실 트래킹."""

    def __init__(
        self,
        start_xy: Tuple[float, float] = (0.0, 0.0),
        start_theta: float = 0.0,
        wheel_diameter: float = 0.10,
        wheel_base: float = 0.30,
    ):
        self.x, self.y = start_xy
        self.theta = start_theta
        self.wheel_diameter = wheel_diameter
        self.wheel_base = wheel_base
        self._tilt_deg = 0.0
        self._fired = 0

    # ── lifecycle ──
    def start(self) -> None:
        print(f"[SIM] robot ready @ ({self.x:.2f}, {self.y:.2f}, "
              f"{np.degrees(self.theta):.1f}°)")

    def stop(self) -> None:
        print(f"[SIM] robot shutdown (fired {self._fired} times)")

    # ── pose source ──
    def get_pose(self) -> Optional[Dict]:
        return {
            "x":           float(self.x),
            "y":           float(self.y),
            "theta":       float(self.theta),
            "theta_deg":   float(np.degrees(self.theta)),
            "tracking":    "OK",
            "tracking_ok": True,
        }

    def is_alive(self) -> bool:
        return True

    def wait_for_tracking(self, timeout: float = 5.0) -> bool:  # noqa: ARG002
        return True

    # ── motor sinks ──
    def send_wheel_omegas(self, omega_left: float, omega_right: float, dt: float) -> None:
        """차동 구동 정기구학으로 즉시 자세 적분."""
        r = self.wheel_diameter / 2.0
        v_L = omega_left * r
        v_R = omega_right * r
        v   = 0.5 * (v_L + v_R)
        w   = (v_R - v_L) / self.wheel_base
        self.x    += v * np.cos(self.theta) * dt
        self.y    += v * np.sin(self.theta) * dt
        self.theta = self._wrap_angle(self.theta + w * dt)

    def tilt_camera(self, deg: float) -> None:
        self._tilt_deg = deg
        print(f"[SIM] camera tilt → {deg:+.1f}°")

    def send_leveling_angles(self, angles_deg, encoder_steps) -> None:
        print(f"[SIM] leveling motors ← deg={[f'{a:+.2f}' for a in angles_deg]}  "
              f"steps={encoder_steps}")

    def fire(self) -> None:
        self._fired += 1
        print(f"[SIM] *** FIRE #{self._fired} ***")

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class RealRobot:
    """실제 RealSense + ORB-SLAM3 측위. 모터는 stub (TODO: 시리얼 드라이버 연결)."""

    def __init__(self, wheel_diameter: float = 0.10, wheel_base: float = 0.30):
        from vio.orbslam_localizer import (  # lazy import
            LocalizerConfig, OrbSlamLocalizer,
        )
        self.wheel_diameter = wheel_diameter
        self.wheel_base = wheel_base
        self.localizer = OrbSlamLocalizer(LocalizerConfig())
        self._fired = 0

    def start(self) -> None:
        self.localizer.start()
        print("[REAL] waiting for SLAM tracking OK ...")
        ok = self.localizer.wait_for_tracking(timeout=30.0)
        print(f"[REAL] tracking_ok = {ok}")
        if not ok:
            raise RuntimeError("ORB-SLAM3 did not reach tracking OK within 30s")

    def stop(self) -> None:
        self.localizer.stop()
        print(f"[REAL] shutdown (fired {self._fired} times)")

    def get_pose(self) -> Optional[Dict]:
        return self.localizer.get_pose()

    def is_alive(self) -> bool:
        return self.localizer.is_alive()

    def wait_for_tracking(self, timeout: float = 30.0) -> bool:
        return self.localizer.wait_for_tracking(timeout=timeout)

    def send_wheel_omegas(self, omega_left: float, omega_right: float, dt: float) -> None:
        # TODO: serial 패킷으로 OpenRB 송신 (Driving/simulation.py 의 SerialCommandSim 참고)
        print(f"\r[REAL] motors ωL={omega_left:+.3f}  ωR={omega_right:+.3f} rad/s",
              end="", flush=True)

    def tilt_camera(self, deg: float) -> None:
        # TODO: 틸트 서보 명령
        print(f"\n[REAL TODO] camera tilt → {deg:+.1f}°")

    def send_leveling_angles(self, angles_deg, encoder_steps) -> None:
        # TODO: 레벨링 모터 컨트롤러 송신
        print(f"[REAL TODO] leveling motors ← deg={angles_deg}  steps={encoder_steps}")

    def fire(self) -> None:
        # TODO: 플라이휠 발사 트리거
        self._fired += 1
        print(f"[REAL TODO] *** FIRE #{self._fired} ***")


# ─────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────
class CapstonePipeline:
    """Phase 1 (Driving) → Phase 2 (Aiming & Strike ×N) 통합 실행기."""

    def __init__(
        self,
        robot,
        target_provider: DummyTargetProvider,
        ctrl: DrivingController,
        ik: LevelingIK,
        dt: float = 0.067,                 # 15 Hz
        phase1_timeout_sec: float = 60.0,
        num_strikes: int = 2,
        strike_interval_sec: float = 1.0,  # 타격 사이 종 진동 대기
    ):
        self.robot = robot
        self.target_provider = target_provider
        self.ctrl = ctrl
        self.ik = ik
        self.dt = dt
        self.phase1_timeout_sec = phase1_timeout_sec
        self.num_strikes = num_strikes
        self.strike_interval_sec = strike_interval_sec

    def run(self) -> bool:
        print("=" * 70)
        print("  Capstone 2026 Pipeline START")
        print("=" * 70)

        ok = self.phase1_driving()
        if not ok:
            print("[PIPELINE] Phase 1 failed → abort")
            return False

        print()
        ok = self.phase2_aiming()
        print()
        print("=" * 70)
        print(f"  Capstone Pipeline {'COMPLETE' if ok else 'FAILED in Phase 2'}")
        print("=" * 70)
        return ok

    # ── Phase 1 ──
    def phase1_driving(self) -> bool:
        target_xy = self.target_provider.get_phase1_target()
        print(f"\n── PHASE 1: DRIVING ──")
        print(f"  target (world) : ({target_xy[0]:+.2f}, {target_xy[1]:+.2f}) m")

        self.ctrl.reset()
        max_steps = int(self.phase1_timeout_sec / self.dt)
        log_every = max(1, int(0.5 / self.dt))   # 0.5s 마다 한 줄 로그

        for step in range(max_steps):
            pose = self.robot.get_pose()
            if pose is None:
                time.sleep(self.dt)
                continue

            if not pose["tracking_ok"]:
                # SLAM lost → 안전 정지
                self.robot.send_wheel_omegas(0.0, 0.0, self.dt)
                if step % log_every == 0:
                    print(f"  [{step*self.dt:5.2f}s]  tracking={pose['tracking']} → STOP")
                time.sleep(self.dt)
                continue

            out = self.ctrl.compute(
                pose["x"], pose["y"], pose["theta"], target_xy[0], target_xy[1])

            self.robot.send_wheel_omegas(
                out["wheel_omega_left"], out["wheel_omega_right"], self.dt)

            if step % log_every == 0:
                print(f"  [{step*self.dt:5.2f}s]  pose=({pose['x']:+.2f}, "
                      f"{pose['y']:+.2f}, {pose['theta_deg']:+6.1f}°)  "
                      f"dist={out['distance']:.2f}  v={out['v']:.2f}  "
                      f"ω_L/R=({out['wheel_omega_left']:+.2f}, "
                      f"{out['wheel_omega_right']:+.2f})")

            if out["reached"]:
                self.robot.send_wheel_omegas(0.0, 0.0, self.dt)
                print(f"  ✓ reached @ t={step*self.dt:.2f}s  "
                      f"(final dist={out['distance']:.3f}m)")
                return True

            time.sleep(self.dt if isinstance(self.robot, RealRobot) else 0.0)

        print(f"  ✗ timeout after {self.phase1_timeout_sec:.0f}s")
        return False

    # ── Phase 2 ──
    def phase2_aiming(self) -> bool:
        print(f"── PHASE 2: AIMING & STRIKE x{self.num_strikes} ──")

        # 카메라 90° 틸트 (위로)
        self.robot.tilt_camera(90.0)
        time.sleep(0.3)

        successful = 0
        for shot in range(1, self.num_strikes + 1):
            print(f"\n  ── shot {shot}/{self.num_strikes} ──")

            # 매 타격 직전 종 위치 재추정 (벨 진동 대응)
            target_xyz = self.target_provider.get_phase2_target()
            print(f"  target (plate frame): ({target_xyz[0]:+.3f}, "
                  f"{target_xyz[1]:+.3f}, {target_xyz[2]:+.3f}) m")

            out = self.ik.aim_at(target_xyz)

            if out["angles_deg"] is None:
                print("  ✗ leg length infeasible — skip")
                continue

            ball = ", ".join(f"{b:.2f}" for b in out["ball_deg"])
            print(f"  motor angles : {[f'{a:+.3f}' for a in out['angles_deg']]} deg")
            print(f"  encoder steps: {out['angles_steps']}")
            print(f"  ball P deg   : [{ball}] (lim={self.ik.cfg.ball_max_deg})")
            print(f"  feasible     : {out['ok']}")

            if not out["ok"]:
                print("  ⚠ ball joint limit exceeded — proceeding anyway "
                      "(real system: nudge mobile base)")

            self.robot.send_leveling_angles(out["angles_deg"], out["angles_steps"])
            time.sleep(0.3)
            self.robot.fire()
            successful += 1

            if shot < self.num_strikes:
                time.sleep(self.strike_interval_sec)

        print(f"\n  → {successful}/{self.num_strikes} strikes executed")
        return successful == self.num_strikes


# ─────────────────────────────────────────────────────────────────────
# Build & main
# ─────────────────────────────────────────────────────────────────────
def build_pipeline(args) -> CapstonePipeline:
    target_cfg = DummyTargetConfig(
        phase1_target=(args.phase1_x, args.phase1_y),
        phase2_target=(args.phase2_x, args.phase2_y, args.phase2_z),
        phase2_jitter=args.phase2_jitter,
    )
    target_provider = DummyTargetProvider(target_cfg)

    ctrl = DrivingController(ControllerConfig(
        wheel_diameter=args.wheel_diameter,
        wheel_base=args.wheel_base,
    ))
    ik = LevelingIK(LevelingConfig())

    if args.mode == "sim":
        robot = SimulatedRobot(
            start_xy=(args.start_x, args.start_y),
            start_theta=np.deg2rad(args.start_theta_deg),
            wheel_diameter=args.wheel_diameter,
            wheel_base=args.wheel_base,
        )
    elif args.mode == "real":
        robot = RealRobot(
            wheel_diameter=args.wheel_diameter,
            wheel_base=args.wheel_base,
        )
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    return CapstonePipeline(
        robot, target_provider, ctrl, ik,
        dt=args.dt,
        phase1_timeout_sec=args.phase1_timeout,
        num_strikes=args.num_strikes,
        strike_interval_sec=args.strike_interval,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Capstone 2026 full-pipeline integration runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=["sim", "real"], default="sim",
                    help="sim: pure-Python integration test (default). "
                         "real: RealSense + ORB-SLAM3 (Pi 환경)")

    # Phase 1 dummy target
    ap.add_argument("--phase1-x", type=float, default=3.0,
                    help="dummy phase1 world target x [m]")
    ap.add_argument("--phase1-y", type=float, default=2.0)

    # Phase 2 dummy target
    ap.add_argument("--phase2-x", type=float, default=0.10,
                    help="dummy phase2 plate-frame target x [m]")
    ap.add_argument("--phase2-y", type=float, default=0.00)
    ap.add_argument("--phase2-z", type=float, default=3.00)
    ap.add_argument("--phase2-jitter", type=float, default=0.05,
                    help="±jitter z noise [m] (vibrating bell)")

    # 차량 기하
    ap.add_argument("--wheel-diameter", type=float, default=0.10)
    ap.add_argument("--wheel-base",     type=float, default=0.30)

    # 시작 자세 (sim 전용)
    ap.add_argument("--start-x", type=float, default=0.0)
    ap.add_argument("--start-y", type=float, default=0.0)
    ap.add_argument("--start-theta-deg", type=float, default=0.0)

    # 루프
    ap.add_argument("--dt", type=float, default=0.067, help="control loop dt [s]")
    ap.add_argument("--phase1-timeout", type=float, default=60.0)
    ap.add_argument("--num-strikes", type=int, default=2)
    ap.add_argument("--strike-interval", type=float, default=1.0)

    args = ap.parse_args()

    pipeline = build_pipeline(args)

    pipeline.robot.start()
    try:
        ok = pipeline.run()
    finally:
        pipeline.robot.stop()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
