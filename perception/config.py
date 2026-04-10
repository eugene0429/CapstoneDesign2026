"""
RealSense D435i Perception 설정 파일
데이터 수집 / VIO 측위 / 타겟 디텍션 통합 설정
"""

import os

# ============================================================
# 카메라 설정
# ============================================================
CAMERA = {
    # 컬러 스트림 해상도 및 FPS
    "color_width": 640,
    "color_height": 480,
    "color_fps": 30,

    # 깊이 스트림 해상도 및 FPS
    "depth_width": 640,
    "depth_height": 480,
    "depth_fps": 30,

    # Depth → Color 정렬 활성화
    "align_depth_to_color": True,

    # IR 이미터 활성화 (스테레오 매칭 품질 향상에 필수)
    "enable_ir_emitter": True,

    # IMU 스트림 활성화 (VIO에서 사용)
    "enable_imu": False,
}

# ============================================================
# 저장 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

PATHS = {
    "dataset": DATASET_DIR,
    "images": os.path.join(DATASET_DIR, "images"),
    "depth": os.path.join(DATASET_DIR, "depth"),
    "videos": os.path.join(DATASET_DIR, "videos"),
    "labels": os.path.join(DATASET_DIR, "labels"),
}

# ============================================================
# 캡처 설정
# ============================================================
CAPTURE = {
    # 자동 캡처 간격 (초)
    "auto_interval": 0.5,

    # 이미지 저장 포맷
    "image_format": ".jpg",
    "image_quality": 95,       # JPEG 품질 (0-100)

    # 깊이 이미지 저장 포맷 (16bit PNG로 원본 깊이값 보존)
    "depth_format": ".png",

    # 비디오 코덱
    "video_codec": "mp4v",
    "video_format": ".mp4",
    "video_fps": 30,
}

# ============================================================
# 시각화 설정
# ============================================================
DISPLAY = {
    "window_name": "RealSense D435i - YOLO Data Capture",
    "show_depth": True,
    "show_info_overlay": True,
    "depth_colormap": 2,       # cv2.COLORMAP_JET = 2
    "font_scale": 0.6,
    "font_color": (0, 255, 0),
    "font_thickness": 1,
}

# ============================================================
# YOLO 데이터셋 구조 설정
# ============================================================
YOLO = {
    # 클래스 목록 (필요에 따라 수정)
    "classes": [
        # "class_0",
        # "class_1",
    ],
    # Train/Val 비율
    "train_ratio": 0.8,
    "val_ratio": 0.2,
}

# ============================================================
# VIO 설정
# ============================================================
VIO = {
    # 특징점 검출기
    "feature_type": "FAST",
    "fast_threshold": 20,
    "max_features": 300,

    # Lucas-Kanade Optical Flow 파라미터
    "lk_win_size": (21, 21),
    "lk_max_level": 3,

    # 키프레임 판별 기준
    "keyframe_min_features": 80,       # 추적 특징점이 이 이하로 떨어지면 키프레임
    "keyframe_max_interval": 10,       # 최대 키프레임 간격 (프레임 수)

    # PnP RANSAC 파라미터
    "pnp_reproj_threshold": 3.0,       # 리프로젝션 오차 임계값 (px)
    "pnp_confidence": 0.99,
    "pnp_min_inliers": 10,             # 최소 인라이어 수

    # 깊이 필터
    "depth_min": 0.3,                  # 최소 깊이 (m)
    "depth_max": 5.0,                  # 최대 깊이 (m)

    # EKF IMU 노이즈 파라미터
    "accel_noise_std": 0.1,            # 가속도계 노이즈 (m/s^2)
    "gyro_noise_std": 0.01,            # 자이로스코프 노이즈 (rad/s)
    "accel_bias_std": 0.01,            # 가속도계 바이어스 랜덤 워크
    "gyro_bias_std": 0.001,            # 자이로 바이어스 랜덤 워크

    # IMU 초기 정렬/정지 상태 억제
    "imu_init_samples": 200,           # 시작 시 정지 상태로 모을 IMU 샘플 수 (100Hz 기준 2초)
    "gravity_magnitude": 9.81,         # 중력 가속도 크기 (m/s^2)
    "stationary_accel_tol": 0.35,      # | |a|-g | 정지 판정 임계값
    "stationary_gyro_tol": 0.08,       # |w| 정지 판정 임계값 (rad/s)
    "zupt_velocity_reset": True,       # 정지 판정 시 속도 0으로 리셋
    "zupt_position_damping": 1.0,      # 정지 판정 시 가속도 적분 억제량
    "zupt_min_frames": 5,              # ZUPT 발동까지 연속 정지 판정 필요 프레임 수 (hysteresis)
    "visual_velocity_alpha": 0.6,      # 비전 기반 속도 추정 반영 비율
    "no_vision_velocity_damping": 0.98,# 비전 보정 실패 시 속도 감쇠

    # 초기 불확실성
    "init_pos_std": 0.01,
    "init_vel_std": 0.1,
    "init_att_std": 0.01,
    "init_bias_std": 0.1,
}

# ============================================================
# 디텍션 설정
# ============================================================
DETECTION = {
    # YOLO 모델 경로
    "model_path": os.path.join(BASE_DIR, "models", "best.pt"),
    # 최소 confidence 임계값
    "confidence_threshold": 0.5,
}
