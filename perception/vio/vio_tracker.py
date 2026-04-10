"""
VIO (Visual-Inertial Odometry) 기반 카메라 측위
================================================

RealSense D435i의 컬러 카메라 + 깊이 + IMU 데이터를 사용하여
카메라의 6DoF 포즈 (위치 + 자세)를 실시간 추정

파이프라인:
  1. FAST 특징점 검출 (키프레임에서)
  2. Lucas-Kanade Optical Flow 추적 (매 프레임)
  3. PnP + RANSAC 포즈 추정 (깊이 활용)
  4. IMU 사전적분 + EKF 융합
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class IMUPreintegrator:
    """IMU 사전적분기 — 두 키프레임 사이의 IMU 측정을 누적"""

    def __init__(self, accel_noise_std, gyro_noise_std):
        self.accel_noise = accel_noise_std
        self.gyro_noise = gyro_noise_std
        self.reset()

    def reset(self):
        self.delta_p = np.zeros(3)  # 위치 변화량
        self.delta_v = np.zeros(3)  # 속도 변화량
        self.delta_R = np.eye(3)    # 회전 변화량
        self.dt_sum = 0.0

    def integrate(self, accel, gyro, dt):
        """단일 IMU 측정값 적분"""
        if dt <= 0 or dt > 0.5:
            return

        accel = np.array(accel)
        gyro = np.array(gyro)

        # 자이로 → 회전 업데이트
        angle = gyro * dt
        angle_norm = np.linalg.norm(angle)
        if angle_norm > 1e-8:
            dR = Rotation.from_rotvec(angle).as_matrix()
        else:
            dR = np.eye(3)

        # 가속도 → 속도/위치 업데이트 (현재 회전 기준)
        accel_world = self.delta_R @ accel
        self.delta_p += self.delta_v * dt + 0.5 * accel_world * dt * dt
        self.delta_v += accel_world * dt
        self.delta_R = self.delta_R @ dR
        self.dt_sum += dt


class EKFState:
    """
    간소화 EKF 상태 벡터 (16차원)
    [position(3), velocity(3), quaternion(4), accel_bias(3), gyro_bias(3)]
    """

    def __init__(self, config):
        self.config = config
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.eye(3)
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)

        # 공분산 행렬 (15x15, 쿼터니언은 에러 상태 3차원 사용)
        self.P = np.diag([
            config["init_pos_std"]**2,  config["init_pos_std"]**2,  config["init_pos_std"]**2,
            config["init_vel_std"]**2,  config["init_vel_std"]**2,  config["init_vel_std"]**2,
            config["init_att_std"]**2,  config["init_att_std"]**2,  config["init_att_std"]**2,
            config["init_bias_std"]**2, config["init_bias_std"]**2, config["init_bias_std"]**2,
            config["init_bias_std"]**2, config["init_bias_std"]**2, config["init_bias_std"]**2,
        ])

        # 프로세스 노이즈
        self.accel_noise = config["accel_noise_std"]
        self.gyro_noise = config["gyro_noise_std"]
        self.accel_bias_noise = config["accel_bias_std"]
        self.gyro_bias_noise = config["gyro_bias_std"]
        self.gravity = np.array([0, config.get("gravity_magnitude", 9.81), 0], dtype=np.float64)

    def predict(self, accel, gyro, dt, stationary=False):
        """IMU 기반 상태 예측 (predict step)"""
        if dt <= 0 or dt > 0.5:
            return

        accel = np.array(accel) - self.accel_bias
        gyro = np.array(gyro) - self.gyro_bias

        # 상태 전이
        accel_world = self.orientation @ accel + self.gravity
        if stationary:
            damping = float(np.clip(self.config.get("zupt_position_damping", 1.0), 0.0, 1.0))
            accel_world *= (1.0 - damping)
            if self.config.get("zupt_velocity_reset", True):
                self.velocity[:] = 0.0
        self.position += self.velocity * dt + 0.5 * accel_world * dt * dt
        self.velocity += accel_world * dt

        # 회전 업데이트
        angle = gyro * dt
        angle_norm = np.linalg.norm(angle)
        if angle_norm > 1e-8:
            dR = Rotation.from_rotvec(angle).as_matrix()
        else:
            dR = np.eye(3)
        self.orientation = self.orientation @ dR

        # 야코비안 F (15x15 상태 전이 행렬)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -self.orientation @ _skew(accel) * dt * dt * 0.5
        F[3:6, 6:9] = -self.orientation @ _skew(accel) * dt
        F[0:3, 9:12] = -self.orientation * dt * dt * 0.5
        F[3:6, 9:12] = -self.orientation * dt
        F[6:9, 12:15] = -np.eye(3) * dt

        # 프로세스 노이즈 Q
        Q = np.zeros((15, 15))
        Q[0:3, 0:3] = np.eye(3) * (self.accel_noise * dt)**2
        Q[3:6, 3:6] = np.eye(3) * (self.accel_noise * dt)**2
        Q[6:9, 6:9] = np.eye(3) * (self.gyro_noise * dt)**2
        Q[9:12, 9:12] = np.eye(3) * (self.accel_bias_noise * dt)**2
        Q[12:15, 12:15] = np.eye(3) * (self.gyro_bias_noise * dt)**2

        self.P = F @ self.P @ F.T + Q

    def correct_pose(self, measured_position, measured_rotation, pos_noise=0.05, rot_noise=0.02):
        """비전 기반 포즈로 EKF correction"""
        # 관측 잔차 (position)
        z_pos = measured_position - self.position

        # 관측 잔차 (rotation) — 에러 쿼터니언에서 소각도 벡터 추출
        R_err = measured_rotation @ self.orientation.T
        rot_vec_err = Rotation.from_matrix(R_err).as_rotvec()

        z = np.concatenate([z_pos, rot_vec_err])

        # 관측 행렬 H (6x15): position(0:3)과 orientation(6:9) 관측
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)   # position
        H[3:6, 6:9] = np.eye(3)   # orientation error

        # 관측 노이즈 R
        R = np.diag([
            pos_noise**2, pos_noise**2, pos_noise**2,
            rot_noise**2, rot_noise**2, rot_noise**2,
        ])

        # 칼만 게인
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 상태 업데이트
        dx = K @ z
        self.position += dx[0:3]
        self.velocity += dx[3:6]

        # 회전 보정
        dtheta = dx[6:9]
        if np.linalg.norm(dtheta) > 1e-10:
            dR = Rotation.from_rotvec(dtheta).as_matrix()
            self.orientation = dR @ self.orientation

        self.accel_bias += dx[9:12]
        self.gyro_bias += dx[12:15]

        # 공분산 업데이트
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form


def _skew(v):
    """3D 벡터의 skew-symmetric 행렬"""
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0],
    ])


def _rotation_between_vectors(src, dst):
    """src 벡터를 dst 벡터로 회전시키는 3x3 회전 행렬"""
    src = np.array(src, dtype=np.float64)
    dst = np.array(dst, dtype=np.float64)

    src_norm = np.linalg.norm(src)
    dst_norm = np.linalg.norm(dst)
    if src_norm < 1e-8 or dst_norm < 1e-8:
        return np.eye(3)

    src = src / src_norm
    dst = dst / dst_norm
    cross = np.cross(src, dst)
    dot = np.clip(np.dot(src, dst), -1.0, 1.0)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-8:
        if dot > 0:
            return np.eye(3)

        axis = np.array([1.0, 0.0, 0.0])
        if abs(src[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - src * np.dot(axis, src)
        axis = axis / np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()

    vx = _skew(cross)
    return np.eye(3) + vx + vx @ vx * ((1.0 - dot) / (cross_norm ** 2))


class VIOTracker:
    """Visual-Inertial Odometry 추적기"""

    def __init__(self, camera_intrinsics, config=None):
        """
        Args:
            camera_intrinsics: RealSense 카메라 내부 파라미터 (rs.intrinsics)
            config: VIO 설정 dict (config.py의 VIO)
        """
        if config is None:
            from config import VIO as _vio_cfg
            config = _vio_cfg

        self.config = config

        # 카메라 매트릭스 구성
        self.fx = camera_intrinsics.fx
        self.fy = camera_intrinsics.fy
        self.cx = camera_intrinsics.ppx
        self.cy = camera_intrinsics.ppy
        self.camera_matrix = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1],
        ], dtype=np.float64)
        self.dist_coeffs = np.array(camera_intrinsics.coeffs, dtype=np.float64)

        # FAST 검출기
        self.detector = cv2.FastFeatureDetector_create(
            threshold=config.get("fast_threshold", 20),
            nonmaxSuppression=True,
        )

        # LK Optical Flow 파라미터
        win = config.get("lk_win_size", (21, 21))
        self.lk_params = dict(
            winSize=win,
            maxLevel=config.get("lk_max_level", 3),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # 상태
        self.pose = np.eye(4)
        self.prev_gray = None
        self.prev_points = None           # 추적 중인 2D 특징점
        self.prev_points_3d = None        # 대응 3D 좌표
        self.frames_since_keyframe = 0
        self.is_initialized = False
        self.prev_timestamp = None

        # EKF
        self.ekf = EKFState(config)

        # IMU 사전적분
        self.imu_preint = IMUPreintegrator(
            config["accel_noise_std"],
            config["gyro_noise_std"],
        )

        self.gravity_magnitude = config.get("gravity_magnitude", 9.81)
        self.imu_init_samples_required = config.get("imu_init_samples", 200)
        self.imu_init_accel_samples = []
        self.imu_init_gyro_samples = []
        self.imu_ready = False
        self.last_visual_position = None
        self.keyframe_pose = np.eye(4)
        self._stationary_count = 0

        # 통계
        self.tracked_count = 0
        self.inlier_count = 0

    def update(self, color_image, depth_image, accel=None, gyro=None, timestamp=None):
        """
        새 프레임으로 포즈 업데이트

        Args:
            color_image: BGR 컬러 이미지
            depth_image: 깊이 이미지 (16bit, 단위: mm)
            accel: (x, y, z) 가속도 데이터 (옵션)
            gyro: (x, y, z) 자이로 데이터 (옵션)
            timestamp: 프레임 타임스탬프 (ms)

        Returns:
            pose: 4x4 변환 행렬 (카메라 → 월드)
        """
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        depth_m = depth_image.astype(np.float32) * 0.001  # mm → m

        # 타임스탬프 dt 계산
        dt = 0.0
        if timestamp is not None and self.prev_timestamp is not None:
            dt = (timestamp - self.prev_timestamp) * 0.001  # ms → s
            if dt < 0 or dt > 1.0:
                dt = 0.0

        accel_cam = None
        gyro_cam = None
        if accel is not None:
            # RealSense IMU (x-right, y-up, z-back) -> Camera Optical (x-right, y-down, z-forward)
            accel_cam = np.array([accel[0], -accel[1], -accel[2]], dtype=np.float64)
        if gyro is not None:
            gyro_cam = np.array([gyro[0], -gyro[1], -gyro[2]], dtype=np.float64)

        # ── IMU predict ──
        if accel_cam is not None and gyro_cam is not None:
            self._accumulate_imu_init(accel_cam, gyro_cam)
            if not self.imu_ready and self._try_initialize_imu():
                self.pose[:3, :3] = self.ekf.orientation
                # 방향이 바뀌었으므로 keyframe 좌표계 무효화 → 다음 프레임에서 새 keyframe 생성
                self.is_initialized = False
                self.prev_points = None

        if accel_cam is not None and gyro_cam is not None and dt > 0 and self.imu_ready:
            if self._is_stationary(accel_cam, gyro_cam):
                self._stationary_count += 1
            else:
                self._stationary_count = 0
            zupt_min = self.config.get("zupt_min_frames", 5)
            stationary = self._stationary_count >= zupt_min
            self.ekf.predict(accel_cam, gyro_cam, dt, stationary=stationary)

        # ── 첫 프레임: 초기화 ──
        if not self.is_initialized:
            if self.imu_ready:
                self.pose[:3, :3] = self.ekf.orientation
                self.pose[:3, 3] = self.ekf.position
            self._init_frame(gray, depth_m)
            self.prev_timestamp = timestamp
            return self.pose.copy()

        # ── Optical Flow 추적 ──
        tracked_2d, tracked_3d, status = self._track_features(gray)
        self.tracked_count = len(tracked_2d) if tracked_2d is not None else 0

        # ── 키프레임 판별 ──
        need_keyframe = self._need_keyframe(tracked_2d)

        # ── 포즈 추정 (추적 성공 시) ──
        if tracked_2d is not None and len(tracked_2d) >= self.config.get("pnp_min_inliers", 10):
            success, R_est, t_est, inliers = self._estimate_pose(tracked_2d, tracked_3d)

            if success:
                self.inlier_count = len(inliers)

                # 비전 포즈 → 4x4 행렬
                T_vision = np.eye(4)
                T_vision[:3, :3] = R_est
                T_vision[:3, 3] = t_est.flatten()

                if accel_cam is not None and gyro_cam is not None and dt > 0 and self.imu_ready:
                    # EKF correction: 비전 관측으로 보정
                    self.ekf.correct_pose(t_est.flatten(), R_est)
                    self._update_velocity_from_vision(t_est.flatten(), dt)
                    self.pose[:3, :3] = self.ekf.orientation
                    self.pose[:3, 3] = self.ekf.position
                else:
                    # IMU 없으면 비전만 사용
                    self.pose = T_vision
                    self.ekf.position = t_est.flatten()
                    self.ekf.orientation = R_est.copy()
                    self._update_velocity_from_vision(t_est.flatten(), dt)
            else:
                # PnP 실패 → IMU 예측만 사용
                if self.imu_ready:
                    self._damp_velocity_without_vision()
                    self.pose[:3, :3] = self.ekf.orientation
                    self.pose[:3, 3] = self.ekf.position
                need_keyframe = True
        else:
            # 추적 실패 → 강제 키프레임
            if self.imu_ready:
                self._damp_velocity_without_vision()
                self.pose[:3, :3] = self.ekf.orientation
                self.pose[:3, 3] = self.ekf.position
            need_keyframe = True

        # ── 키프레임 업데이트 ──
        if need_keyframe:
            self._init_frame(gray, depth_m)
        else:
            self.prev_gray = gray
            self.prev_points = tracked_2d
            self.prev_points_3d = tracked_3d
            self.frames_since_keyframe += 1

        self.prev_timestamp = timestamp
        return self.pose.copy()

    def _accumulate_imu_init(self, accel_cam, gyro_cam):
        """초기 정지 구간 IMU 평균값 수집 (정지 상태 확인 포함)"""
        if self.imu_ready:
            return

        # 정지 상태 확인: 가속도 크기가 중력에서 많이 벗어나거나 자이로가 크면 움직이는 것
        accel_norm = np.linalg.norm(accel_cam)
        gyro_norm = np.linalg.norm(gyro_cam)
        if abs(accel_norm - self.gravity_magnitude) > 1.5 or gyro_norm > 0.3:
            # 움직임 감지: 수집된 샘플 초기화
            self.imu_init_accel_samples.clear()
            self.imu_init_gyro_samples.clear()
            return

        self.imu_init_accel_samples.append(accel_cam.copy())
        self.imu_init_gyro_samples.append(gyro_cam.copy())

        max_keep = max(self.imu_init_samples_required, 1)
        if len(self.imu_init_accel_samples) > max_keep:
            self.imu_init_accel_samples = self.imu_init_accel_samples[-max_keep:]
        if len(self.imu_init_gyro_samples) > max_keep:
            self.imu_init_gyro_samples = self.imu_init_gyro_samples[-max_keep:]

    def _try_initialize_imu(self):
        """초기 IMU 평균으로 자세/바이어스 초기화"""
        if self.imu_ready or len(self.imu_init_accel_samples) < self.imu_init_samples_required:
            return False

        accel_samples = np.array(self.imu_init_accel_samples)
        gyro_samples = np.array(self.imu_init_gyro_samples)

        # 분산 확인: 샘플이 너무 불안정하면 초기화 거부 후 재수집
        accel_std = np.std(accel_samples, axis=0)
        if np.max(accel_std) > 0.4:
            self.imu_init_accel_samples.clear()
            self.imu_init_gyro_samples.clear()
            return False

        accel_mean = np.mean(accel_samples, axis=0)
        gyro_mean = np.mean(gyro_samples, axis=0)
        accel_norm = np.linalg.norm(accel_mean)
        if accel_norm < 1e-6:
            return False

        target_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        self.ekf.orientation = _rotation_between_vectors(accel_mean, target_up)
        self.ekf.gyro_bias = gyro_mean

        expected_specific_force = accel_mean / accel_norm * self.gravity_magnitude
        self.ekf.accel_bias = accel_mean - expected_specific_force
        self.imu_ready = True
        return True

    def _is_stationary(self, accel_cam, gyro_cam):
        """바이어스 보정 후 IMU가 정지 상태인지 판별"""
        accel_unbiased = accel_cam - self.ekf.accel_bias
        gyro_unbiased = gyro_cam - self.ekf.gyro_bias
        accel_norm = np.linalg.norm(accel_unbiased)
        gyro_norm = np.linalg.norm(gyro_unbiased)

        accel_tol = self.config.get("stationary_accel_tol", 0.35)
        gyro_tol = self.config.get("stationary_gyro_tol", 0.08)
        return (
            abs(accel_norm - self.gravity_magnitude) < accel_tol
            and gyro_norm < gyro_tol
        )

    def _update_velocity_from_vision(self, measured_position, dt):
        """비전 위치 변화량으로 속도 상태를 다시 고정"""
        measured_position = np.array(measured_position, dtype=np.float64)
        if self.last_visual_position is not None and dt > 1e-4:
            visual_velocity = (measured_position - self.last_visual_position) / dt
            alpha = float(np.clip(self.config.get("visual_velocity_alpha", 0.6), 0.0, 1.0))
            self.ekf.velocity = (1.0 - alpha) * self.ekf.velocity + alpha * visual_velocity
        self.last_visual_position = measured_position.copy()

    def _damp_velocity_without_vision(self):
        """비전이 잠시 끊길 때 관성으로 계속 밀리는 현상 완화"""
        damping = float(np.clip(self.config.get("no_vision_velocity_damping", 0.98), 0.0, 1.0))
        self.ekf.velocity *= damping

    def _init_frame(self, gray, depth_m):
        """키프레임 초기화: 새 특징점 검출 + 3D 좌표 계산"""
        keypoints = self.detector.detect(gray)

        if len(keypoints) == 0:
            return

        # 최대 특징점 수 제한 (응답 기준 정렬)
        max_feat = self.config.get("max_features", 300)
        if len(keypoints) > max_feat:
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_feat]

        points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        # 깊이 유효한 특징점만 필터링
        valid_2d, valid_3d = self._filter_with_depth(points_2d, depth_m)

        if len(valid_2d) >= self.config.get("pnp_min_inliers", 10):
            self.prev_gray = gray
            self.prev_points = valid_2d
            self.prev_points_3d = valid_3d
            self.keyframe_pose = self.pose.copy()
            self.last_visual_position = None  # keyframe 전환 시 cross-keyframe 속도 계산 방지
            self.frames_since_keyframe = 0
            self.is_initialized = True

    def _filter_with_depth(self, points_2d, depth_m):
        """2D 특징점 중 유효한 깊이를 가진 것만 선택하고 3D 좌표 계산"""
        d_min = self.config.get("depth_min", 0.3)
        d_max = self.config.get("depth_max", 5.0)

        valid_2d = []
        valid_3d = []

        h, w = depth_m.shape
        for pt in points_2d:
            u, v = int(round(pt[0])), int(round(pt[1]))
            if 0 <= u < w and 0 <= v < h:
                z = depth_m[v, u]
                if d_min < z < d_max:
                    x = (pt[0] - self.cx) * z / self.fx
                    y = (pt[1] - self.cy) * z / self.fy
                    valid_2d.append(pt)
                    valid_3d.append([x, y, z])

        return (
            np.array(valid_2d, dtype=np.float32).reshape(-1, 2) if valid_2d else np.array([], dtype=np.float32),
            np.array(valid_3d, dtype=np.float64).reshape(-1, 3) if valid_3d else np.array([], dtype=np.float64),
        )

    def _track_features(self, gray):
        """LK Optical Flow로 이전 특징점 추적"""
        if self.prev_points is None or len(self.prev_points) == 0:
            return None, None, None

        pts = self.prev_points.reshape(-1, 1, 2)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, pts, None, **self.lk_params
        )

        if next_pts is None:
            return None, None, status

        status = status.flatten().astype(bool)

        # 역방향 검증 (forward-backward check)
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, next_pts, None, **self.lk_params
        )
        if back_pts is not None:
            back_status = back_status.flatten().astype(bool)
            fb_dist = np.linalg.norm(
                pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
            )
            fb_good = fb_dist < 1.0  # 1px 이내
            status = status & back_status & fb_good

        tracked_2d = next_pts.reshape(-1, 2)[status]
        tracked_3d = self.prev_points_3d[status]

        return tracked_2d, tracked_3d, status

    def _need_keyframe(self, tracked_2d):
        """키프레임이 필요한지 판별"""
        if tracked_2d is None:
            return True

        min_feat = self.config.get("keyframe_min_features", 80)
        max_interval = self.config.get("keyframe_max_interval", 10)

        if len(tracked_2d) < min_feat:
            return True
        if self.frames_since_keyframe >= max_interval:
            return True

        return False

    def _estimate_pose(self, points_2d, points_3d):
        """PnP + RANSAC으로 카메라 포즈 추정"""
        if len(points_2d) < 4:
            return False, None, None, None

        # 3D 좌표를 keyframe 당시의 포즈로 월드 프레임 변환
        # self.pose 대신 keyframe_pose 사용: 프레임마다 pose가 바뀌어도 3D점 기준이 흔들리지 않음
        R_kf = self.keyframe_pose[:3, :3]
        t_kf = self.keyframe_pose[:3, 3]
        world_3d = (R_kf @ points_3d.T).T + t_kf

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_3d.astype(np.float64),
            points_2d.astype(np.float64),
            self.camera_matrix,
            self.dist_coeffs,
            reprojectionError=self.config.get("pnp_reproj_threshold", 3.0),
            confidence=self.config.get("pnp_confidence", 0.99),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None:
            return False, None, None, None

        if len(inliers) < self.config.get("pnp_min_inliers", 10):
            return False, None, None, None

        R_cam, _ = cv2.Rodrigues(rvec)
        t_cam = tvec.flatten()

        # solvePnP는 world→camera 변환을 줌 → camera→world로 역변환
        R_world = R_cam.T
        t_world = -R_cam.T @ t_cam

        return True, R_world, t_world.reshape(3, 1), inliers.flatten()

    def get_position(self):
        """현재 위치 반환 (x, y, z) 미터"""
        return self.pose[:3, 3].copy()

    def get_rotation(self):
        """현재 회전 행렬 반환 (3x3)"""
        return self.pose[:3, :3].copy()

    def get_pose(self):
        """현재 6DoF 포즈 반환 (4x4 변환 행렬)"""
        return self.pose.copy()

    def get_euler_degrees(self):
        """현재 자세를 오일러각 (roll, pitch, yaw) 도 단위로 반환"""
        r = Rotation.from_matrix(self.pose[:3, :3])
        return r.as_euler('xyz', degrees=True)

    def get_stats(self):
        """추적 상태 통계 반환"""
        return {
            'tracked_features': self.tracked_count,
            'inliers': self.inlier_count,
            'frames_since_keyframe': self.frames_since_keyframe,
            'initialized': self.is_initialized,
        }

    def reset(self):
        """포즈 초기화"""
        self.pose = np.eye(4)
        self.prev_gray = None
        self.prev_points = None
        self.prev_points_3d = None
        self.frames_since_keyframe = 0
        self.is_initialized = False
        self.prev_timestamp = None
        self.ekf = EKFState(self.config)
        self.imu_preint.reset()
        self.imu_init_accel_samples.clear()
        self.imu_init_gyro_samples.clear()
        self.imu_ready = False
        self.last_visual_position = None
        self.keyframe_pose = np.eye(4)
        self._stationary_count = 0
        self.tracked_count = 0
        self.inlier_count = 0
