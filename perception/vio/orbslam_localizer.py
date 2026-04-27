"""
ORB-SLAM3 Localization Module — production library.

`main.py orbslam --pi --no-imu --headless` 의 동작을 모듈 형태로 분리.
파이프라인이 import 해서 world-frame (x, y, θ) pose 를 stream 으로 받아 쓸 수 있다.

사용 예
------
    from vio.orbslam_localizer import OrbSlamLocalizer, LocalizerConfig
    from controller import DrivingController, ControllerConfig

    ctrl = DrivingController(ControllerConfig(wheel_diameter=0.10, wheel_base=0.30))

    with OrbSlamLocalizer() as loc:
        loc.wait_for_tracking(timeout=30.0)
        while True:
            pose = loc.get_pose()
            if pose is None or not pose["tracking_ok"]:
                continue
            out = ctrl.compute(pose["x"], pose["y"], pose["theta"], tx, ty)
            if out["reached"]:
                break
            send_to_motors(out["wheel_omega_left"], out["wheel_omega_right"])

월드 좌표계 (카메라 출발 자세 = origin)
--------------------------------------
    world_x =  camera_z   (forward)
    world_y = -camera_x   (left)
    theta   = yaw         (CCW positive viewed from above) [rad]
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# 기존 헬퍼 재사용 (yaml 빌드, 캘리브레이션 캐시, 경로 상수 등)
from vio.orbslam_runner import (
    BINARY_IMU,
    BINARY_NO_IMU,
    CONFIG_IMU,
    CONFIG_NO_IMU,
    CONFIG_PI,
    ORBSLAM3_DIR,
    STATE_NAMES,
    VOCAB,
    _flush_realsense,
    _load_cached_calibration,
    _save_calibration_cache,
    build_yaml,
    get_camera_calibration,
)


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
@dataclass
class LocalizerConfig:
    # ── 동작 모드 (main.py 의 --pi --no-imu --headless 와 동일이 default) ──
    use_imu: bool = False        # False == --no-imu
    pi_mode: bool = True         # True  == --pi   (RGB-D 424×240@15fps + nFeatures=500 yaml)

    # ── 시작 / 종료 ──
    max_retries: int = 3                # subprocess 기동 실패 시 재시도 횟수
    startup_settle_sec: float = 1.0     # 기동 직후 즉시 크래시 감지를 위한 대기
    term_timeout_sec: float = 5.0       # stop() 시 SIGTERM → SIGKILL 전환 대기

    # ── tracking 대기 ──
    tracking_poll_sec: float = 0.05     # wait_for_tracking() 폴링 주기

    # ── 메모리 ──
    trajectory_cap: int = 5000          # 저장 trajectory 점 상한 (초과 시 절반 폐기)


# ──────────────────────────────────────────────
# Localizer
# ──────────────────────────────────────────────
class OrbSlamLocalizer:
    """ORB-SLAM3 기반 측위 모듈 (헤드리스, 라이브러리 API)."""

    def __init__(self, cfg: Optional[LocalizerConfig] = None):
        self.cfg = cfg if cfg is not None else LocalizerConfig()
        self._proc: Optional[subprocess.Popen] = None
        self._tmp_dir: Optional[str] = None
        self._stop_evt: Optional[threading.Event] = None
        self._stdout_f = None
        self._stderr_f = None

        self._lock = threading.Lock()
        self._latest_raw: Optional[np.ndarray] = None   # 최신 [x,y,z,qx,qy,qz,qw]
        self._tracking_state: int = -1                  # NOT_READY
        self._all_positions: List[np.ndarray] = []      # 카메라-프레임 (x,y,z) 누적

    # ── context manager ──
    def __enter__(self) -> "OrbSlamLocalizer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ── lifecycle ──
    def start(self) -> None:
        """ORB-SLAM3 subprocess 기동. 즉시 크래시 시 cfg.max_retries 까지 재시도."""
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            _flush_realsense()
            try:
                self._spawn_once()
            except Exception as e:
                last_err = e
                self._cleanup_subprocess()
                time.sleep(1.0)
                continue

            # settle 후에도 살아있는지 확인
            time.sleep(self.cfg.startup_settle_sec)
            if self.is_alive():
                return
            print(f"[ORBSLAM] subprocess died on startup "
                  f"({attempt+1}/{self.cfg.max_retries}), retrying...")
            self._cleanup_subprocess()
            time.sleep(1.0)

        msg = f"ORB-SLAM3 startup failed after {self.cfg.max_retries} attempts"
        if last_err is not None:
            msg += f": {last_err}"
        raise RuntimeError(msg)

    def stop(self) -> None:
        """Subprocess 종료 + reader 스레드 정지 + 임시 파일 정리."""
        if self._stop_evt is not None:
            self._stop_evt.set()
        self._cleanup_subprocess()

    # ── internal: subprocess setup ──
    def _spawn_once(self) -> None:
        c = self.cfg
        binary = BINARY_IMU if c.use_imu else BINARY_NO_IMU
        base_cfg = (CONFIG_IMU if c.use_imu
                    else (CONFIG_PI if c.pi_mode else CONFIG_NO_IMU))

        for path, name in [(binary, "binary"), (VOCAB, "Vocabulary"),
                           (base_cfg, "base yaml")]:
            if not os.path.exists(path):
                raise RuntimeError(f"{name} not found: {path}")

        # 캘리브레이션 (캐시 우선)
        calib = _load_cached_calibration()
        if not calib:
            cal_w, cal_h, cal_fps = (640, 480, 15) if c.pi_mode else (640, 480, 30)
            calib = get_camera_calibration(width=cal_w, height=cal_h, fps=cal_fps,
                                           use_imu=c.use_imu)
            if calib:
                _save_calibration_cache(calib)

        # tmp yaml
        self._tmp_dir = tempfile.mkdtemp(prefix="orbslam_")
        config_path = os.path.join(self._tmp_dir, "RealSense_D435i_calib.yaml")
        if calib:
            build_yaml(calib, base_cfg, config_path)
        else:
            shutil.copy(base_cfg, config_path)

        # env (헤드리스 + ORB-SLAM3 lib path)
        env = os.environ.copy()
        env["ORBSLAM_NO_VIEWER"] = "1"
        libs = [
            os.path.join(ORBSLAM3_DIR, "lib"),
            os.path.join(ORBSLAM3_DIR, "Thirdparty/DBoW2/lib"),
            os.path.join(ORBSLAM3_DIR, "Thirdparty/g2o/lib"),
        ]
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(libs + ([existing] if existing else []))

        stdout_path = os.path.join(self._tmp_dir, "stdout.log")
        stderr_path = os.path.join(self._tmp_dir, "stderr.log")
        self._stdout_f = open(stdout_path, "w")
        self._stderr_f = open(stderr_path, "w")

        self._proc = subprocess.Popen(
            [binary, VOCAB, config_path],
            stdout=self._stdout_f,
            stderr=self._stderr_f,
            env=env,
        )

        # reader 스레드 (파일 tail 방식 — 현재 process 가 파일에 쓰는 동안 읽음)
        self._stop_evt = threading.Event()
        threading.Thread(target=self._tail_pose_lines,
                         args=(stdout_path,), daemon=True).start()
        threading.Thread(target=self._tail_state_lines,
                         args=(stderr_path,), daemon=True).start()

    def _cleanup_subprocess(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=self.cfg.term_timeout_sec)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

        for f in (self._stdout_f, self._stderr_f):
            try:
                if f is not None and not f.closed:
                    f.close()
            except Exception:
                pass
        self._stdout_f = None
        self._stderr_f = None

        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        self._tmp_dir = None

        # C++ 가 남기는 frame jpg
        frame_jpg = "/tmp/orbslam_frame.jpg"
        if os.path.exists(frame_jpg):
            try:
                os.remove(frame_jpg)
            except OSError:
                pass

    # ── internal: log readers ──
    def _tail_pose_lines(self, path: str) -> None:
        with open(path, "r") as f:
            while self._stop_evt is not None and not self._stop_evt.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.01)
                    continue
                line = line.strip()
                if not line.startswith("POSE:"):
                    continue
                try:
                    vals = list(map(float, line.split()[1:]))
                except ValueError:
                    continue
                if len(vals) != 7:
                    continue
                arr = np.array(vals)
                with self._lock:
                    self._latest_raw = arr
                    self._all_positions.append(arr[:3].copy())
                    cap = self.cfg.trajectory_cap
                    if len(self._all_positions) > cap:
                        self._all_positions = self._all_positions[cap // 2:]

    def _tail_state_lines(self, path: str) -> None:
        with open(path, "r") as f:
            while self._stop_evt is not None and not self._stop_evt.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                line = line.rstrip()
                if not line.startswith("STATE:"):
                    continue
                try:
                    state = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    continue
                with self._lock:
                    self._tracking_state = state

    # ── public API ──
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def is_tracking(self) -> bool:
        with self._lock:
            return self._tracking_state == 2  # OK

    def get_tracking_state(self) -> str:
        with self._lock:
            s = self._tracking_state
        return STATE_NAMES.get(s, f"?({s})")

    def get_pose(self) -> Optional[Dict]:
        """
        World-frame pose 반환.

        Returns
        -------
        dict | None
            x            : float   world X [m]   (= camera Z)
            y            : float   world Y [m]   (= -camera X)
            theta        : float   yaw [rad]     (CCW positive)
            theta_deg    : float   yaw [deg]
            tracking     : str     'NOT_READY' | 'NO_IMAGE' | 'INIT' | 'OK'
                                  | 'RECENTLY_LOST' | 'LOST'
            tracking_ok  : bool    (tracking == 'OK')
            raw          : np.ndarray (7,) [x, y, z, qx, qy, qz, qw] (camera frame)

        한 번도 POSE 라인이 들어오지 않았으면 None.
        """
        with self._lock:
            raw = self._latest_raw
            state = self._tracking_state
        if raw is None:
            return None

        # 카메라 프레임 → 월드 프레임
        # world_x =  camera_z (forward), world_y = -camera_x (left)
        cam_pos = raw[:3]
        quat = raw[3:7]  # scipy: [x, y, z, w]

        from scipy.spatial.transform import Rotation
        try:
            R = Rotation.from_quat(quat).as_matrix()
        except Exception:
            return None

        forward = R[:, 2]   # camera +z 방향이 월드에서 가리키는 방향
        theta_rad = float(math.atan2(-forward[0], forward[2]))

        return {
            "x":           float(cam_pos[2]),
            "y":           float(-cam_pos[0]),
            "theta":       theta_rad,
            "theta_deg":   math.degrees(theta_rad),
            "tracking":    STATE_NAMES.get(state, f"?({state})"),
            "tracking_ok": state == 2,
            "raw":         raw.copy(),
        }

    def wait_for_tracking(self, timeout: float = 30.0) -> bool:
        """tracking_state == OK 가 될 때까지 대기. 성공 True / 타임아웃·dead False."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.is_alive():
                return False
            if self.is_tracking():
                return True
            time.sleep(self.cfg.tracking_poll_sec)
        return False

    def get_trajectory(self) -> np.ndarray:
        """누적 카메라-프레임 위치 (N, 3). 디버깅·로깅용."""
        with self._lock:
            return np.array(self._all_positions) if self._all_positions \
                else np.empty((0, 3))

    def get_total_distance(self) -> float:
        traj = self.get_trajectory()
        if len(traj) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


# ──────────────────────────────────────────────
# CLI: 라이브러리 단독 동작 확인
# ──────────────────────────────────────────────
def _print_loop(cfg: LocalizerConfig) -> None:
    """main.py 의 orbslam --headless 와 동등한 터미널 출력 루프."""
    print(f"[Localizer] starting "
          f"(IMU={'ON' if cfg.use_imu else 'OFF'}, "
          f"Pi={'ON' if cfg.pi_mode else 'OFF'}) — Ctrl+C to stop")
    print(f"{'state':>14s}  {'x_m':>8s}  {'y_m':>8s}  {'theta_deg':>10s}")
    with OrbSlamLocalizer(cfg) as loc:
        try:
            while loc.is_alive():
                pose = loc.get_pose()
                if pose is None:
                    state = loc.get_tracking_state()
                    print(f"\r{state:>14s}  {'--':>8s}  {'--':>8s}  {'--':>10s}",
                          end="", flush=True)
                else:
                    print(f"\r{pose['tracking']:>14s}  "
                          f"{pose['x']:8.3f}  {pose['y']:8.3f}  "
                          f"{pose['theta_deg']:10.2f}",
                          end="", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n[Localizer] Ctrl+C — shutting down")
        finally:
            n = len(loc.get_trajectory())
            print(f"\n[Localizer] {n} poses, {loc.get_total_distance():.3f} m traveled")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="ORB-SLAM3 localization module — headless library demo.")
    ap.add_argument("--imu",   action="store_true",
                    help="enable IMU (default: off, == --no-imu)")
    ap.add_argument("--no-pi", action="store_true",
                    help="disable Pi-optimized yaml (default: pi mode on)")
    args = ap.parse_args()

    _print_loop(LocalizerConfig(
        use_imu=args.imu,
        pi_mode=not args.no_pi,
    ))
