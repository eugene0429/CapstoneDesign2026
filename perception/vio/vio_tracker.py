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

    def predict(self, accel, gyro, dt):
        """IMU 기반 상태 예측 (predict step)"""
        if dt <= 0 or dt > 0.5:
            return

        accel = np.array(accel) - self.accel_bias
        gyro = np.array(gyro) - self.gyro_bias

        GRAVITY = np.array([0, 0, -9.81])

        # 상태 전이
        accel_world = self.orientation @ accel + GRAVITY
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

        # ── IMU predict ──
        if accel is not None and gyro is not None and dt > 0:
            self.ekf.predict(accel, gyro, dt)

        # ── 첫 프레임: 초기화 ──
        if not self.is_initialized:
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

                if accel is not None and gyro is not None and dt > 0:
                    # EKF correction: 비전 관측으로 보정
                    self.ekf.correct_pose(t_est.flatten(), R_est)
                    self.pose[:3, :3] = self.ekf.orientation
                    self.pose[:3, 3] = self.ekf.position
                else:
                    # IMU 없으면 비전만 사용
                    self.pose = T_vision
                    self.ekf.position = t_est.flatten()
                    self.ekf.orientation = R_est.copy()
            else:
                # PnP 실패 → IMU 예측만 사용
                if accel is not None:
                    self.pose[:3, :3] = self.ekf.orientation
                    self.pose[:3, 3] = self.ekf.position
                need_keyframe = True
        else:
            # 추적 실패 → 강제 키프레임
            if accel is not None:
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

        # 3D 좌표를 현재 월드 프레임으로 변환
        R_prev = self.pose[:3, :3]
        t_prev = self.pose[:3, 3]
        world_3d = (R_prev @ points_3d.T).T + t_prev

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
        self.tracked_count = 0
        self.inlier_count = 0
