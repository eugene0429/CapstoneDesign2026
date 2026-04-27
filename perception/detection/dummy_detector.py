"""
Dummy Target Provider — YOLO 학습 전 파이프라인 통합 테스트용.

실제 detection 파이프라인 (YOLO + depth + frame averaging + camera→world 변환)
을 우회해서, Phase 1 / Phase 2 의 타겟 좌표를 *직접* 반환하는 stub.

YOLO 학습이 끝나면 [perception/detection/detector.py](detector.py) +
[perception/detection/position_estimator.py](position_estimator.py) 의 실제
체인으로 교체. 그 시점에는 본 클래스가 더 이상 필요 없다.

좌표계
------
- get_phase1_target()  : world frame (x, y) [m]
                         로봇 출발 자세 = origin, world_x = camera forward.
- get_phase2_target()  : plate-base frame (x, y, z) [m]
                         플레이트 중심 (0, 0, H0=Lc) 에서 본 종 위치.
                         (실제 시스템에서는 Camera→Plate 외부 변환이 적용된 값)

진동하는 종 시뮬레이션
--------------------
get_phase2_target() 는 매 호출 시 phase2_jitter (m) 만큼의 z 노이즈를
더해 반환한다 (3 m 높이에서 수직 진동하는 종을 모사). 이를 통해
"매 타격 직전 3D 벡터 재추정" 로직이 정상 동작하는지 확인 가능.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DummyTargetConfig:
    # ── Phase 1: 종 베이스의 지면 투영 좌표 (world frame) ──
    phase1_target: Tuple[float, float] = (3.0, 2.0)        # (x, y) [m]

    # ── Phase 2: 플레이트 중심 → 종까지의 3D 벡터 ──
    phase2_target: Tuple[float, float, float] = (0.10, 0.00, 3.00)  # (x, y, z) [m]
    phase2_jitter: float = 0.05                            # ±jitter 의 z 노이즈 [m]
    phase2_jitter_seed: int = 42

    # ── Phase 1 multi-frame 평균 모사 (선택) ──
    phase1_noise_std: float = 0.0                          # 단일 검출 노이즈 std [m]
    phase1_avg_frames: int = 1                             # 평균에 사용할 프레임 수


class DummyTargetProvider:
    """파이프라인 통합 테스트용 타겟 좌표 제공기."""

    def __init__(self, cfg: DummyTargetConfig | None = None):
        self.cfg = cfg if cfg is not None else DummyTargetConfig()
        self._rng = np.random.default_rng(self.cfg.phase2_jitter_seed)

    # ── Phase 1 ──
    def get_phase1_target(self) -> Tuple[float, float]:
        """
        출발 직전, YOLO 다중 프레임 평균으로 추정된 종의 world (x, y).

        실제 구현에서는:
          for _ in range(N): bbox = yolo.detect(frame); xyz = depth_deproj(bbox);
          xy_world = T_world_cam @ xyz   →   평균
        """
        c = self.cfg
        if c.phase1_noise_std <= 0 or c.phase1_avg_frames <= 1:
            return c.phase1_target

        # 다중 프레임 평균 시뮬레이션
        samples = np.array([
            (c.phase1_target[0] + self._rng.normal(0, c.phase1_noise_std),
             c.phase1_target[1] + self._rng.normal(0, c.phase1_noise_std))
            for _ in range(c.phase1_avg_frames)
        ])
        avg = samples.mean(axis=0)
        return (float(avg[0]), float(avg[1]))

    # ── Phase 2 ──
    def get_phase2_target(self) -> Tuple[float, float, float]:
        """
        플레이트 중심 기준 종까지의 3D 벡터 (매 호출 시 z 진동 jitter 적용).

        실제 구현에서는:
          bbox  = yolo.detect(frame_after_tilt)
          xyz_c = depth_deproject(bbox)
          xyz_p = T_plate_cam @ xyz_c        # 카메라 → 플레이트 외부 변환
        """
        c = self.cfg
        x, y, z = c.phase2_target
        if c.phase2_jitter > 0:
            z = z + float(self._rng.uniform(-c.phase2_jitter, c.phase2_jitter))
        return (float(x), float(y), float(z))
